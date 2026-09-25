"""End-to-end entity resolution pipeline: normalize -> block (TF-IDF cosine) -> features -> LightGBM -> outputs.

  python -m src.pipeline normalize --split train        # pools -> work/pools/
  python -m src.pipeline normalize --split test
  python -m src.pipeline train  --n-entities 100000     # blocking + features + LightGBM + threshold tuning
  python -m src.pipeline predict                        # test -> output/matching_results.tsv, candidate_pairs.tsv

Country is an open set of labels: it is used only to partition blocking, never as a model feature.
On Linux, worker processes share the fitted TF-IDF matrices through fork (copy-on-write).
"""
import argparse, glob, gc, json, os, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from multiprocessing import get_context
from sklearn.feature_extraction.text import TfidfVectorizer
from .norm import norm_name, norm_addr
from .features import pair_features, add_group_features

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd=lambda p,**k: pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False,quoting=3,**k)
SRCS=('s2','s3'); _G={}          # fork-shared state for workers

# ----------------------------------------------------------------------------- paths
def paths(a):
    d=dict(train=a.train_dir,test=a.test_dir,work=a.work_dir)
    d['pools']=os.path.join(a.work_dir,'pools'); return d

# ----------------------------------------------------------------------------- normalize
def _norm_chunk(c):
    c=c.copy(); c['nn']=c.business_name.map(norm_name); c['na']=[norm_addr(x,k) for x,k in zip(c.business_address,c.country)]
    return c.rename(columns={'business_name':'name','business_address':'addr'})[['entity_id','country','name','addr','nn','na']]

def normalize_file(path,out_prefix,n_jobs,chunk=200000):
    schema=pa.schema([('entity_id',pa.string()),('name',pa.string()),('addr',pa.string()),('nn',pa.string()),('na',pa.string())])
    writers={}; n=0; t0=time.time()
    ctx=get_context('fork' if hasattr(os,'fork') else 'spawn')
    with ctx.Pool(n_jobs) as pool:
        for c in pool.imap(_norm_chunk,rd(path,chunksize=chunk)):
            for k,g in c.groupby('country'):
                if k not in writers: writers[k]=pq.ParquetWriter(f'{out_prefix}_{k}.parquet',schema)
                writers[k].write_table(pa.Table.from_pandas(g[['entity_id','name','addr','nn','na']],schema=schema,preserve_index=False))
            n+=len(c); print(os.path.basename(path),n,round(time.time()-t0),'s',flush=True)
    for w in writers.values(): w.close()

def cmd_normalize(a):
    P=paths(a); os.makedirs(P['pools'],exist_ok=True)
    base=P[a.split]; pre='train_' if a.split=='train' else 'test_'
    for i in ('1','2','3'):
        normalize_file(os.path.join(base,f'{pre}source{i}.tsv'),os.path.join(P['pools'],f'{a.split}_s{i}'),a.n_jobs)

def load_pool(pools,split,src,country):
    f=os.path.join(pools,f'{split}_{src}_{country}.parquet')
    return pd.read_parquet(f) if os.path.exists(f) else None

# ----------------------------------------------------------------------------- blocking
def topk_cosine(Q,Tt,k,chunk=50):
    lens=[];ids=[];vals=[]
    for i in range(0,Q.shape[0],chunk):
        S=(Q[i:i+chunk]@Tt).tocsr()
        for r in range(S.shape[0]):
            a,b=S.indptr[r],S.indptr[r+1]; ix=S.indices[a:b]; v=S.data[a:b]
            if len(v)>k: p=np.argpartition(-v,k)[:k]; ix,v=ix[p],v[p]
            lens.append(len(v)); ids.append(ix); vals.append(v)
    if not ids: return np.zeros(0,int),np.zeros(0,int),np.zeros(0,np.float32)
    return np.array(lens),np.concatenate(ids),np.concatenate(vals)

def build_retriever(pools,maxdf):
    R={}
    for src,pool in pools.items():
        for ch,col in (('name','nn'),('addr','na')):
            vec=TfidfVectorizer(token_pattern=r"\S+",lowercase=False,sublinear_tf=True,max_df=maxdf,dtype=np.float32)
            try: T=vec.fit_transform(pool[col].values)
            except ValueError: continue          # empty vocabulary
            R[(src,ch)]=(vec,T.T.tocsr())
    return R

def make_pairs(qdf,pools,R,K):
    """Union of the name-channel and address-channel top-K by cosine, per source. Returns DataFrame of raw fields."""
    frames={}
    for (src,ch),(vec,Tt) in R.items():
        Q=vec.transform(qdf['nn' if ch=='name' else 'na'].values)
        lens,ix,v=topk_cosine(Q,Tt,K)
        frames[(src,ch)]=pd.DataFrame({'q':np.repeat(np.arange(len(qdf)),lens),'src':src,'c':ix,ch+'_cos':v})
    parts=[]
    for src in pools:
        n=frames.get((src,'name')); a=frames.get((src,'addr'))
        if n is None and a is None: continue
        if n is None: p=a.assign(name_cos=np.nan)
        elif a is None: p=n.assign(addr_cos=np.nan)
        else: p=n.merge(a,on=['q','src','c'],how='outer')
        parts.append(p)
    if not parts: return pd.DataFrame()
    P=pd.concat(parts,ignore_index=True)
    P['in_name']=P.name_cos.notna().astype(int); P['in_addr']=P.addr_cos.notna().astype(int)
    P[['name_cos','addr_cos']]=P[['name_cos','addr_cos']].fillna(0.0)
    b=[]
    for src,pool in pools.items():
        m=P[P.src==src]
        if len(m)==0: continue
        r=pool.iloc[m.c.values]; b.append(pd.DataFrame({'b_name':r.name.values,'b_addr':r.addr.values,'cand_id':r.entity_id.values},index=m.index))
    P=P.join(pd.concat(b))
    P['a_name']=qdf.name.values[P.q.values]; P['a_addr']=qdf.addr.values[P.q.values]; P['a_country']=qdf.country.values[P.q.values]
    P['b_country']=P.a_country      # blocking is per country
    P['s1_id']=qdf.entity_id.values[P.q.values]
    return P

def featurize(P):
    F=pair_features(P); F=add_group_features(F,P[['q','src']]); F['src_is_s3']=(P.src=='s3').astype(int).values
    return F

# ----------------------------------------------------------------------------- shard workers
def _shard(i):
    G=_G; qdf=G['q'].iloc[i:i+G['shard']].reset_index(drop=True)
    P=make_pairs(qdf,G['pools'],G['R'],G['K'])
    out=os.path.join(G['out'],f"{G['tag']}_{i}.parquet")
    if len(P)==0:
        pd.DataFrame({'s1_id':qdf.entity_id,'cands':'','matches':''}).to_parquet(out); return out
    F=featurize(P)
    if G['mode']=='train':
        truth=G['truth']; F=F.astype(np.float32)
        F['s1_id']=P.s1_id.values; F['cand_id']=P.cand_id.values; F['src']=P.src.values
        F['label']=[int(c in truth.get(s,())) for s,c in zip(P.s1_id,P.cand_id)]
        F.to_parquet(out); return out
    p=G['model'].predict(F[G['cols']].values,num_threads=1)
    d=pd.DataFrame({'s1_id':P.s1_id.values,'cand_id':P.cand_id.values,'keep':p>=G['thr']})
    cands=d.groupby('s1_id').cand_id.agg(','.join); mt=d[d.keep].groupby('s1_id').cand_id.agg(','.join)
    r=pd.DataFrame({'s1_id':qdf.entity_id.values}); r['cands']=r.s1_id.map(cands).fillna(''); r['matches']=r.s1_id.map(mt).fillna('')
    r.to_parquet(out); return out

def run_country(country,q1,split,a,mode,tag,out,extra):
    pools={s:load_pool(paths(a)['pools'],split,s,country) for s in SRCS}; pools={s:p for s,p in pools.items() if p is not None and len(p)}
    if not pools:
        f=os.path.join(out,f'{tag}_empty.parquet'); pd.DataFrame({'s1_id':q1.entity_id.values,'cands':'','matches':''}).to_parquet(f); return [f]
    t=time.time(); R=build_retriever(pools,a.maxdf); print(country,'retriever fit',round(time.time()-t),'s',{s:len(p) for s,p in pools.items()},flush=True)
    _G.clear(); _G.update(q=q1.reset_index(drop=True),pools=pools,R=R,K=a.k,shard=a.shard,out=out,tag=tag,mode=mode,**extra)
    starts=list(range(0,len(q1),a.shard)); files=[]
    if a.n_jobs>1 and hasattr(os,'fork'):
        with get_context('fork').Pool(a.n_jobs) as pool:
            for k,f in enumerate(pool.imap_unordered(_shard,starts)): files.append(f); print(country,f'shard {k+1}/{len(starts)}',round(time.time()-t),'s',flush=True)
    else:
        for k,i in enumerate(starts): files.append(_shard(i)); print(country,f'shard {k+1}/{len(starts)}',round(time.time()-t),'s',flush=True)
    _G.clear(); del R,pools; gc.collect(); return files

# ----------------------------------------------------------------------------- metric
def f05(pred,tru):
    if not pred and not tru: return 1.0
    if not pred or not tru: return 0.0
    h=len(pred&tru); p,r=h/len(pred),h/len(tru)
    return 0.0 if h==0 else 1.25*p*r/(0.25*p+r)

# ----------------------------------------------------------------------------- train
def cmd_train(a):
    import lightgbm as lgb
    P=paths(a); out=os.path.join(a.work_dir,'train_parts'); os.makedirs(out,exist_ok=True)
    gt=rd(os.path.join(P['train'],'train_ground_truth.tsv')).sample(a.n_entities,random_state=0)
    truth={s:set(x for x in m.split(',') if x) for s,m in zip(gt.source1_entity_id,gt.matched_entity_ids)}
    need=set(truth); s1=pd.concat([c[c.entity_id.isin(need)] for c in rd(os.path.join(P['train'],'train_source1.tsv'),chunksize=500000)])
    s1=_norm_chunk(s1).reset_index(drop=True); files=[]
    for c in s1.country.unique():
        files+=run_country(c,s1[s1.country==c],'train',a,'train',f'tr_{c}',out,dict(truth=truth))
    D=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    ents=np.array(sorted(need)); rng=np.random.RandomState(0); rng.shuffle(ents); n=len(ents)
    split={**{e:'train' for e in ents[:int(.6*n)]},**{e:'val' for e in ents[int(.6*n):int(.8*n)]},**{e:'test' for e in ents[int(.8*n):]}}
    D['split']=D.s1_id.map(split); cols=[c for c in D.columns if c not in('s1_id','cand_id','src','label','split')]
    tr,va,te=[D.split==s for s in ('train','val','test')]
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.03,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=a.n_jobs,verbose=-1)
    m.fit(D.loc[tr,cols],D.label[tr],eval_set=[(D.loc[va,cols],D.label[va])],callbacks=[lgb.early_stopping(100,verbose=False)])
    D['p']=m.predict_proba(D[cols])[:,1]
    def macro(name,thr):
        d=D[D.split==name]; sel=d[d.p>=thr].groupby('s1_id').cand_id.apply(set)
        return np.mean([f05(sel.get(e,set()),truth[e]) for e in ents if split[e]==name])
    grid=np.arange(0.2,0.96,0.05); v=[macro('val',t) for t in grid]; thr=float(grid[int(np.argmax(v))])
    print('threshold',round(thr,2),'val F0.5',round(max(v),4),'test F0.5',round(macro('test',thr),4),flush=True)
    m.booster_.save_model(os.path.join(a.work_dir,'matcher.txt'))
    json.dump(dict(threshold=thr,columns=cols,k=a.k,maxdf=a.maxdf),open(os.path.join(a.work_dir,'matcher_meta.json'),'w'))

# ----------------------------------------------------------------------------- predict
def cmd_predict(a):
    import lightgbm as lgb
    P=paths(a); out=os.path.join(a.work_dir,'pred_parts'); os.makedirs(out,exist_ok=True)
    meta=json.load(open(os.path.join(a.work_dir,'matcher_meta.json'))); model=lgb.Booster(model_file=os.path.join(a.work_dir,'matcher.txt'))
    s1=_norm_chunk(rd(os.path.join(P['test'],'test_source1.tsv'))).reset_index(drop=True); order=s1.entity_id.values; files=[]
    for c in s1.country.unique():
        if a.only_country and c!=a.only_country: continue
        q1=s1[s1.country==c]
        if a.limit: q1=q1.head(a.limit)
        files+=run_country(c,q1,'test',a,'predict',f'te_{c}',out,dict(model=model,cols=meta['columns'],thr=meta['threshold']))
    R=pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates('s1_id').set_index('s1_id')
    R=R.reindex(order).fillna('')
    os.makedirs(a.output_dir,exist_ok=True)
    pd.DataFrame({'source1_entity_id':order,'matched_entity_ids':R.matches.values}).to_csv(os.path.join(a.output_dir,'matching_results.tsv'),sep='\t',index=False,quoting=3)
    pd.DataFrame({'source1_entity_id':order,'candidate_entity_ids':R.cands.values}).to_csv(os.path.join(a.output_dir,'candidate_pairs.tsv'),sep='\t',index=False,quoting=3)
    print('wrote',len(order),'rows;',(R.matches!='').mean().round(3),'share with >=1 match')

# ----------------------------------------------------------------------------- cli
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['normalize','train','predict'])
    ds=os.path.join(ROOT,'Hackathon','Datasets')
    ap.add_argument('--train-dir',default=os.path.join(ds,'Train Datasets')); ap.add_argument('--test-dir',default=ds)
    ap.add_argument('--work-dir',default=os.path.join(ROOT,'work')); ap.add_argument('--output-dir',default=os.path.join(ROOT,'output'))
    ap.add_argument('--split',default='train',choices=['train','test']); ap.add_argument('--n-jobs',type=int,default=max(1,os.cpu_count()-1))
    ap.add_argument('--n-entities',type=int,default=20000); ap.add_argument('--k',type=int,default=20); ap.add_argument('--maxdf',type=int,default=20000)
    ap.add_argument('--shard',type=int,default=2000); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--only-country',default='')
    a=ap.parse_args(); {'normalize':cmd_normalize,'train':cmd_train,'predict':cmd_predict}[a.cmd](a)

if __name__=='__main__': main()
