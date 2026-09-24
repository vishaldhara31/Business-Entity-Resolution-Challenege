import pandas as pd, numpy as np
D='D:/Amazon_ML_Challenge/Hackathon/Datasets/Train Datasets/'
out='D:/Amazon_ML_Challenge/work/'
g=pd.read_parquet(out+'dev_gt.parquet')
ids=set(i for l in g.matched_entity_ids for i in l.split(',') if i)
rng=np.random.RandomState(1)
pos=[];neg=[]
for c in pd.read_csv(D+'train_source3.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,chunksize=250000):
    m=c.entity_id.isin(ids).values
    pos.append(c[m]); o=c[~m]
    neg.append(o.iloc[np.flatnonzero(rng.rand(len(o))<0.02)])  # ~2% distractor sample
pos=pd.concat(pos); neg=pd.concat(neg)
if len(neg)>len(pos)*3: neg=neg.sample(len(pos)*3,random_state=1)
x=pd.concat([pos,neg]); x.to_parquet(out+'dev_s3.parquet'); print(len(pos),len(x))
print(pos.sample(6,random_state=2).to_string())
