#!/usr/bin/env python


import itertools, subprocess, pathlib, yaml, uuid, json, sys, re, sqlite3, time
import pandas as pd
from tqdm import tqdm


DEBUG = False

SPACE = {
    "mixup_alpha":   [0.8, 1.0, 1.2]            if not DEBUG else [0.8],
    "randaug_choice":["2_9","2_14","3_15"]      if not DEBUG else ["2_9"],
    "warmup":        [5, 10],
    "wd":            [2e-4, 5e-4, 8e-4]         if not DEBUG else [2e-4, 5e-4],
}
EPOCHS  = (30, 60, 100) if not DEBUG else (3, 9, 12)
DB_NAME = "grid30_60_100.db" if not DEBUG else "debug.db"

ROOT = pathlib.Path(__file__).resolve().parent
TMP  = ROOT / "yaml" / "stage_tmp"; TMP.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "train_cifar100.py"
VAL_RE = re.compile(r"val_acc_epoch(\d+)=(\d+\.\d+)")

BASE_CFG = {
    "model":"resnet101", "batch":128, "lr":0.1,
    "amp":True, "seed":42, "optimizer":"SGD", "momentum":0.9,
    "channels_last":False, "compile":False, "ema":None
}


def combos(space):
    keys, vals = zip(*space.items())
    for v in itertools.product(*vals):
        yield dict(zip(keys, v))

def init_db(path):
    con = sqlite3.connect(path)
    con.execute("DROP TABLE IF EXISTS trials")
    con.execute("""CREATE TABLE trials(
        number INTEGER, stage INTEGER, state TEXT,
        params TEXT, epoch INTEGER, val REAL, duration REAL)""")
    return con


def run_train(cfg, epochs, bar):
    yml = TMP / f"{uuid.uuid4().hex[:6]}.yaml"
    yaml.safe_dump(cfg, yml.open("w"))
    t0  = time.time()
    proc = subprocess.Popen([sys.executable, str(TRAIN),
                             "--cfg", str(yml), "--epochs", str(epochs)],
                             stdout=subprocess.PIPE, text=True, encoding="utf-8")
    v = 0.0
    for line in proc.stdout:
        m = VAL_RE.search(line)
        if m:
            ep, v = int(m.group(1)), float(m.group(2))
            bar.set_postfix(epoch=ep, acc=f"{v:.3f}", refresh=False)
    proc.wait()
    return v, time.time() - t0


def stage(stage_id, cand, budget, keep_frac, con, overal_bar, trial_map):

    results=[]
    with tqdm(cand, desc=f"S{stage_id}+{budget}ep", ncols=90) as bar:
        for p in bar:
            num = trial_map.setdefault(tuple(p.items()), len(trial_map))

            scheduler = "cosine_warm" if p["warmup"] > 0 else "cosine"

            cfg = BASE_CFG | {
                "mixup": {"alpha": p["mixup_alpha"]},
                "randaug": {"num_ops": int(p["randaug_choice"].split('_')[0]),
                            "magnitude": int(p["randaug_choice"].split('_')[1])},
                "warmup": p["warmup"],
                "wd":     p["wd"],
                "scheduler": scheduler,
                "outdir": f"runs/S{stage_id}_"+"_".join(map(str, p.values()))
            }
            val, dur = run_train(cfg, budget, bar)
            con.execute("INSERT INTO trials VALUES (?,?,?,?,?,?,?)",
                        (num, stage_id, "COMPLETE", json.dumps(p),
                         budget, val, dur))
            con.commit()
            results.append((p, val))
            overal_bar.update(1)
    k = max(1, int(len(results) * keep_frac))
    return [p for p, _ in sorted(results, key=lambda x: x[1], reverse=True)[:k]]


def main():
    cand0      = list(combos(SPACE))
    n_total    = len(cand0) + len(cand0)//3 + len(cand0)//9
    con        = init_db(DB_NAME)
    trial_map  = {}
    t0         = time.time()

    with tqdm(total=n_total, desc="Total", ncols=90) as total_bar:
        cand1 = stage(1, cand0,  EPOCHS[0], 1/3, con, total_bar, trial_map)   # 30
        cand2 = stage(2, cand1,  EPOCHS[1], 1/3, con, total_bar, trial_map)   # 60
        stage(3,  cand2,  EPOCHS[2], 1.0, con, total_bar, trial_map)          # 100


    df = pd.read_sql("SELECT * FROM trials", con)

    param_cols = ["mixup_alpha", "randaug_choice", "warmup", "wd"]
    params_exp = df["params"].apply(json.loads).apply(pd.Series)[param_cols]
    df_exp = pd.concat([df.drop(columns="params"), params_exp], axis=1)


    (df_exp.rename(columns={"val":"val_acc"})
          [["number","state","epoch","val_acc","duration",*param_cols]]
          .to_csv("grid_intermediate.csv", index=False))


    final = (df_exp.sort_values("epoch")
                     .groupby("number").last()
                     .reset_index())
    final["state"] = "COMPLETE"
    final = final.rename(columns={"val":"value"})
    (final[["number","state","value","duration",*param_cols]]
          .to_csv("grid_trials.csv", index=False))


    df_exp.to_csv("stage30_60_100_summary.csv", index=False)


    best = final.nlargest(1, "value").iloc[0]
    best_params = {k: (best[k].item() if hasattr(best[k], "item") else best[k])
                   for k in param_cols}
    print("\n=== BEST ===")
    print(json.dumps(best_params, indent=2),
          f"val_acc={best.value:.4f}")
    print(f"Elapsed {(time.time()-t0)/3600:.1f} h "
          f"({'DEBUG' if DEBUG else 'FULL'})")

if __name__ == "__main__":
    main()
