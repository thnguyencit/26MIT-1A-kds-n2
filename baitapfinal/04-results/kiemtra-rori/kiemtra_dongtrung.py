import gzip, json, sys, pandas as pd
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in gzip.open(f,'rt')]
    df=pd.DataFrame(rows).astype(str)
    k=df.user_id+"|"+df.item_id+"|"+df["size"]
    d=df[k.duplicated(keep=False)]
    full=df.duplicated(keep=False).sum()
    print(f.split('/')[-1], "pair-dup rows", len(d), "fully identical rows", full)
    cols=[c for c in df.columns if c in ("review_text","review_date","date","fit")]
    g=d.assign(k=k[d.index]).groupby("k")
    same_text=sum(x["review_text"].nunique()==1 for _,x in g) if "review_text" in df else None
    print(" groups", g.ngroups, "groups with identical review_text", same_text)
