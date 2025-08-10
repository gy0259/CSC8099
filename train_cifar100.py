#!/usr/bin/env python

import os, json, time, datetime, argparse, yaml, random, shutil, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch import amp
from torchmetrics.classification import MulticlassCalibrationError
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ---------- utils ----------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_tag(cfg: dict) -> str:

    tags = []
    if cfg.get("mixup"):
        tags.append(f"mix{cfg['mixup']['alpha']}")
    if cfg.get("randaug"):
        ra = cfg["randaug"]
        tags.append(f"ra{ra['num_ops']}-{ra['magnitude']}")
    if cfg.get("warmup", 0):
        tags.append(f"warm{cfg['warmup']}")
    tags.append(f"wd{cfg['wd']}")
    return "_".join(tags)

@torch.no_grad()
def evaluate_full(net, loader, crit, device, ece_metric=None):
    net.eval(); tot = hit = loss_sum = 0
    logits_lst, labels_lst = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = net(x)
        loss_sum += crit(out, y).item() * y.size(0)
        hit      += (out.argmax(1) == y).sum().item(); tot += y.size(0)
        if ece_metric is not None:
            logits_lst.append(out); labels_lst.append(y)
    ece = None
    if ece_metric and logits_lst:
        prob = torch.cat(logits_lst).softmax(1)
        ece  = ece_metric(prob, torch.cat(labels_lst)).item()
    return loss_sum / tot, hit / tot, ece

def plot_curve(loss, acc, path):
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(loss, label='Train Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.grid()
    ax2 = ax1.twinx(); ax2.plot(acc, label='Val Acc', color='g'); ax2.set_ylabel('Acc')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

# ---------- MixUp ----------
def apply_mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], (y, y[idx], lam)

def mixup_loss(crit, preds, tgt):
    y_a, y_b, lam = tgt
    return lam * crit(preds, y_a) + (1 - lam) * crit(preds, y_b)

 ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, help='YAML config')
    ap.add_argument('--batch', type=int, help='override batch')
    ap.add_argument('--epochs', type=int, help='override epochs')
    ap.add_argument('--outdir', default='runs')
    args = ap.parse_args()

     ---- default cfg ----
    cfg = {
        'model': 'resnet101', 'epochs': 100, 'batch': 128, 'lr': 0.1, 'wd': 5e-4,
        'amp': True, 'seed': 42, 'scheduler': 'cosine', 'warmup': 0,
        'warmup_iters': None, 'mixup': None, 'ema': None, 'randaug': None,
        'channels_last': False, 'compile': False
    }

     ---- YAML  ----
    if args.cfg:
        with open(args.cfg, 'r', encoding='utf-8') as f:
            cfg.update(yaml.safe_load(f) or {})
        cfg['_cfg_file'] = os.path.abspath(args.cfg)  # 记录 YAML 路径
    if args.batch:   cfg['batch']  = args.batch
    if args.epochs is not None: cfg['epochs'] = args.epochs
    cfg['outdir'] = args.outdir

    ---- dirs ----
    ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    param_tag = make_tag(cfg)
    run_dir   = os.path.join(cfg['outdir'], f"{cfg['model']}_{param_tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

     ----  cfg  ----
    with open(os.path.join(run_dir, 'run_cfg.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if args.cfg:
        shutil.copy2(args.cfg, os.path.join(run_dir, 'config.yaml'))

    # ---- env & metric init ----
    set_seed(cfg['seed'])
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = cfg['amp'] and device == 'cuda'
    ece_metric = MulticlassCalibrationError(num_classes=100, n_bins=15).to(device)

    # ----  cfg ----
    print('\n=== Effective Config ===')
    print(json.dumps(cfg, indent=2, ensure_ascii=False), '\n')

    # ---- data ----
    tf_train = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip()]
    if cfg['randaug']:
        tf_train.insert(0, transforms.RandAugment(cfg['randaug']['num_ops'],
                                                  cfg['randaug']['magnitude']))
    tf_train += [transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)]
    tfm_train = transforms.Compose(tf_train)
    tfm_val   = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,)*3, (0.5,)*3)])

    tr_loader = DataLoader(datasets.CIFAR100('./data', True,  download=True, transform=tfm_train),
                           batch_size=cfg['batch'], shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(datasets.CIFAR100('./data', False, download=True, transform=tfm_val),
                           batch_size=cfg['batch']*2, shuffle=False, num_workers=4, pin_memory=True)

    # ---- model ----
    model = models.resnet101(num_classes=100, weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); model.maxpool = nn.Identity()
    model = model.to(device)
    if cfg['compile'] and torch.__version__ >= "2.0":
        model = torch.compile(model)

    # ---- optim & sched ----
    optim = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=cfg['wd'])
    if cfg['scheduler'] == 'cosine_warm' and cfg['warmup'] > 0:
        warm_iters = cfg['warmup_iters'] if cfg['warmup_iters'] else cfg['warmup']
        warm   = LinearLR(optim, start_factor=0.1, total_iters=warm_iters)
        cosine = CosineAnnealingLR(optim, T_max=cfg['epochs']-warm_iters)
        sched  = SequentialLR(optim, [warm, cosine], [warm_iters])
    else:
        sched = CosineAnnealingLR(optim, T_max=cfg['epochs'])

    criterion = nn.CrossEntropyLoss()
    scaler    = amp.GradScaler(enabled=use_amp)
    writer    = SummaryWriter(run_dir + '/tb')

    # ---- loop ----
    rows, t0, best_acc = [], time.time(), 0.0
    use_mix = cfg.get('mixup') is not None and 'alpha' in cfg['mixup']

    for ep in range(cfg['epochs']):
        model.train()
        ep_loss = ep_hit = ep_tot = 0
        tic = time.time()

        for x, y in tqdm(tr_loader, desc=f'E{ep+1}/{cfg["epochs"]}', ncols=80):
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                if use_mix:
                    x_mix, tgt = apply_mixup(x, y, cfg['mixup']['alpha'])
                    preds = model(x_mix)
                    loss  = mixup_loss(criterion, preds, tgt)
                else:
                    preds = model(x)
                    loss  = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            ep_loss += loss.item() * y.size(0)
            if not use_mix:
                ep_hit += (preds.argmax(1) == y).sum().item()
                ep_tot += y.size(0)

        train_loss = ep_loss / len(tr_loader.dataset)
        train_acc  = (ep_hit / ep_tot) if not use_mix else float('nan')

        val_loss, val_acc, val_ece = evaluate_full(
            model, va_loader, criterion, device, ece_metric
        )
        sched.step(); cur_lr = optim.param_groups[0]['lr']
        best_acc = max(best_acc, val_acc)
        print(f"val_acc_epoch{ep + 1}={val_acc:.4f}")

        # ---- TensorBoard ----
        writer.add_scalar('Loss/train', train_loss, ep)
        writer.add_scalar('Loss/val',   val_loss,   ep)
        if not use_mix:
            writer.add_scalar('Acc/train', train_acc, ep)
        writer.add_scalar('Acc/val',    val_acc,    ep)
        writer.add_scalar('ECE/val',    val_ece,    ep)
        writer.add_scalar('LR',         cur_lr,     ep)

        row = {
            'epoch': ep+1,
            'train_loss': train_loss,
            'train_acc' : train_acc,
            'val_loss'  : val_loss,
            'val_acc'   : val_acc,
            'val_ece'   : val_ece,
            'lr'        : cur_lr,
            'ips'       : len(tr_loader.dataset)/(time.time()-tic),
            'mem_GB'    : torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else 0,
            'time_s'    : time.time()-tic
        }
        row.update(cfg)
        rows.append(row)

        print(f"E{ep+1}: train_loss {train_loss:.3f} | "
              f"val_acc {val_acc:.2%} | ECE {val_ece:.4f}")
        if ep + 1 == cfg['epochs']:
            print(f"val_acc={val_acc:.4f}")

        
        print(f"epoch={ep + 1} val_acc={val_acc:.4f}")

    # ---- save ----
    plot_curve([r['train_loss'] for r in rows],
               [r['val_acc']    for r in rows],
               f'{run_dir}/curve_{ts}.png')
    pd.DataFrame(rows).to_csv(f'{run_dir}/metrics_{ts}.csv', index=False)
    pd.DataFrame({'total_time_s':[time.time()-t0]}).to_csv(
        f'{run_dir}/summary_{ts}.csv', index=False
    )
    torch.save(model.state_dict(), f'{run_dir}/final.pth')

    try:
        from scripts.export_env import export_env; export_env(run_dir)
    except ModuleNotFoundError:
        pass

    print(f'[DONE] best Acc {best_acc:.2%}; total {(time.time()-t0)/60:.1f} min')

if __name__ == '__main__':
    main()
