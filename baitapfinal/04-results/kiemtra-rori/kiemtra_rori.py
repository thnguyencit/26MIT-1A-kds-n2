import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location("d", sys.argv[1]+"/sizerec/data.py"); d = importlib.util.module_from_spec(spec); sys.modules["d"]=d; spec.loader.exec_module(d)
for f in sys.argv[2:]:
    data = d.load_dataset(f); df = data.df
    print("==", data.name, len(df), "cols:", list(df.columns), "body:", data.body_cols)
    print(" leaky in df/body:", [c for c in d.LEAKY_FIELDS if c in df.columns or c in data.body.columns])
    print(" size median", df["size"].median(), "top sizes", df["size"].value_counts().head(5).to_dict())
    key = df.user_id+"|"+df.item_size
    print(" dup (user,item,size) rows:", int(key.duplicated(keep=False).sum()), "groups:", int(key[key.duplicated()].nunique()))
    for proto in ["cold","warm"]:
        ov=[]; dup=[]; agree=[]
        for seed in range(1,11):
            tr,va,te = d.split(df, proto, seed)
            assert len(set(tr)&set(te))==0 and len(set(tr)&set(va))==0 and len(set(va)&set(te))==0
            assert len(tr)+len(va)+len(te)==len(df)
            ut=set(df.user_id.iloc[tr]); ue=set(df.user_id.iloc[te]); uv=set(df.user_id.iloc[va])
            ov.append((len(ut&ue), len(ut&uv)))
            ktr=set(key.iloc[tr]); kte=key.iloc[te]
            m=kte.isin(ktr); dup.append(m.mean())
            if m.any():
                lab=df.iloc[tr].assign(k=key.iloc[tr]).groupby("k")["y"].agg(lambda s:s.mode()[0])
                agree.append((df.y.iloc[te][m.values].values==lab.loc[kte[m].values].values).mean())
        print(f" {proto}: user overlap train-test/train-val (seed1..3) {ov[:3]}; test rows whose (user,item,size) also in train: mean {np.mean(dup)*100:.2f}%", (f"label agree {np.mean(agree)*100:.1f}%" if agree else ""), " sizes test:", len(te))
