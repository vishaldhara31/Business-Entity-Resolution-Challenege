import nbformat as nbf
nb=nbf.v4.new_notebook()
C=lambda s:nbf.v4.new_code_cell(s.strip('\n')); M=lambda s:nbf.v4.new_markdown_cell(s.strip('\n'))
nb.cells=[
M("# Entity matching: S1 -> S2/S3\nDev sample (200k S1). Steps: load, normalize, block, measure candidate recall."),
C('''
import os, re, unicodedata
os.environ['TMP']=os.environ['TEMP']='D:/tmp'
import pandas as pd, numpy as np
from collections import defaultdict
W='D:/Amazon_ML_Challenge/work/'
s1=pd.read_parquet(W+'dev_s1.parquet'); s2=pd.read_parquet(W+'dev_s2.parquet'); s3=pd.read_parquet(W+'dev_s3.parquet')
gt=pd.read_parquet(W+'dev_gt.parquet')
print(s1.shape,s2.shape,s3.shape); s1.head(3)
'''),
M("## Normalization"),
C('''
SUFFIX=r"\b(private|pvt|limited|ltd|llp|l l p|llc|l l c|inc|incorporated|corp|corporation|co|company|the|and)\b"
def norm_name(x):
    x=unicodedata.normalize('NFKD',str(x)).lower()
    x=re.sub(r"^(india|us|usa)\s{2,}","",x)               # country prefix seen in S3
    x=re.sub(r"\.(com|in|net|org|co\.in)\b","",x)         # domain-style names
    x=re.sub(r"[^a-z0-9 ]"," ",x)
    x=re.sub(SUFFIX," ",x)
    return re.sub(r"\s+"," ",x).strip()
def toks(x): return norm_name(x).split()
for d in (s1,s2,s3): d['nn']=d.business_name.map(norm_name)
s1[['business_name','nn']].sample(5,random_state=0)
'''),
M("## Blocking: shared rare name tokens (same country)\nIndex each S2/S3 row by its tokens; candidate = rows sharing a token with document frequency <= cap."),
C('''
CAP=200
def build_index(d):
    idx=defaultdict(list)
    for i,(n,c) in enumerate(zip(d.nn,d.country)):
        for t in set(n.split()): idx[(c,t)].append(i)
    return {k:v for k,v in idx.items() if len(v)<=CAP}
def candidates(q,idx,d):
    out=set()
    for n,c in zip(q.nn,q.country):
        pass
    return out
def cand_pairs(q,d,idx):
    res=[]
    for qi,(n,c) in enumerate(zip(q.nn,q.country)):
        s=set()
        for t in set(n.split()): s.update(idx.get((c,t),()))
        res.append(s)
    return res
idx2,idx3=build_index(s2),build_index(s3)
q=s1.sample(20000,random_state=0).reset_index(drop=True)
c2,c3=cand_pairs(q,s2,idx2),cand_pairs(q,s3,idx3)
print('avg cands',np.mean([len(s) for s in c2]),np.mean([len(s) for s in c3]))
'''),
M("## Candidate recall vs ground truth"),
C('''
g=gt.set_index('source1_entity_id').matched_entity_ids.str.split(',')
def recall(cs,d,prefix):
    eid=d.entity_id.values; hit=tot=0
    for qid,s in zip(q.entity_id,cs):
        truth={i for i in g[qid] if i.startswith(prefix)}
        got={eid[j] for j in s}
        hit+=len(truth&got); tot+=len(truth)
    return hit/tot
print('S2 recall',recall(c2,s2,'S2'),'S3 recall',recall(c3,s3,'S3'))
'''),
]
nb.metadata['kernelspec']={'name':'amzml','display_name':'Python (amzml)','language':'python'}
nbf.write(nb,'notebooks/01_blocking.ipynb')
