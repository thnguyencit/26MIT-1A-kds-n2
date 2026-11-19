import sys, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from sizerec.data import load_dataset
from sizerec.experiment import run_experiment
from sizerec.trainer import HParams
import torch; torch.set_num_threads(int(sys.argv[4]))
f, seeds = sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
models = ["mf","mf_body","body_product","mf_body_product"]
data = load_dataset(f)
out=[]
for tag in ["orig","dedup"]:
    d = data
    if tag=="dedup":
        import gzip, json
        raw = pd.DataFrame([json.loads(l) for l in gzip.open(f,'rt')])
        raw = raw[raw["fit"].isin(["small","fit","large"])].reset_index(drop=True)
        keep = ~raw.astype(str).duplicated().values
        d.df = data.df[keep].reset_index(drop=True); d.body = data.body[keep].reset_index(drop=True)
        print("dedup rows", len(d.df), flush=True)
    res, sig, sz = run_experiment(d, ["warm"], seeds, models, HParams(), log=lambda s: print(tag, s, flush=True), n_boot=200)
    r = res[res.bucket.isin(["all","2-4"])].groupby(["bucket","model"])["auc"].mean().rename(tag)
    out.append(r)
    print(tag, "SIG", sig[sig.bucket.isin(["all","2-4","5+"])][["bucket","vs","delta_auc","ci95_low","ci95_high","p_bootstrap","p_ttest"]].round(4).to_string(), flush=True)
    print(tag, "SUMMARY", res.groupby(["bucket","model"])["auc"].agg(["mean","std"]).round(4).to_string(), flush=True)
    s2 = sz.groupby("method")[["exact","within_1"]].mean(); print(tag, s2, flush=True)
import pickle; pickle.dump((out,), open(sys.argv[5],"wb"))
print(pd.concat(out,axis=1).assign(diff=lambda x:x.dedup-x.orig).round(4))
