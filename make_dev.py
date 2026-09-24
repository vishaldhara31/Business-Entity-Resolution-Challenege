import pandas as pd, numpy as np
D='D:/Amazon_ML_Challenge/Hackathon/Datasets/Train Datasets/'
r=lambda f:pd.read_csv(D+f,sep='\t',dtype=str,keep_default_na=False,quoting=3)
g=r('train_ground_truth.tsv').sample(200000,random_state=0)
s1=r('train_source1.tsv'); s1=s1[s1.entity_id.isin(g.source1_entity_id)]
print(s1.country.value_counts().to_dict())
g=g.set_index('source1_entity_id').loc[s1.entity_id]
print('singleton by country',(g.matched_entity_ids.values=='').mean())
ids=set(i for l in g.matched_entity_ids for i in l.split(',') if i)
out='D:/Amazon_ML_Challenge/work/'
s1.to_parquet(out+'dev_s1.parquet'); g.reset_index().to_parquet(out+'dev_gt.parquet')
for k in ('2','3'):
    d=r(f'train_source{k}.tsv'); pos=d[d.entity_id.isin(ids)]
    mask=~d.entity_id.isin(ids).values; idx=np.random.RandomState(1).choice(np.flatnonzero(mask),len(pos)*3,replace=False); neg=d.iloc[np.sort(idx)]  # distractors
    x=pd.concat([pos,neg]); x.to_parquet(out+f'dev_s{k}.parquet'); print(k,len(pos),len(x))
    print(pos.sample(8,random_state=2).to_string())

