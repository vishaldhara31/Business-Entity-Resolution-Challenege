"""Country-agnostic pair features for the matcher (no country one-hot, no fixed country list)."""
import re
import numpy as np, pandas as pd
from functools import lru_cache
from rapidfuzz import fuzz
from .norm import norm_name, norm_addr

_SYN={'private':'pvt','limited':'ltd','corporation':'corp','incorporated':'inc','company':'co'}
_LEGAL={'pvt','ltd','llp','llc','inc','corp','co','pbc','pc','plc','lp','sarl','sas','sasu','eurl','sa','sci','snc'}
_DOMAIN=re.compile(r"\.(com|in|net|org|io|biz|info)\b|^@|www\.",re.I)
_TOK=re.compile(r"[^\W_]+")

nn_c=lru_cache(maxsize=500000)(norm_name)
na_c=lru_cache(maxsize=500000)(norm_addr)

def _legal(raw):
    return frozenset(_SYN.get(t,t) for t in _TOK.findall(str(raw).lower()) if _SYN.get(t,t) in _LEGAL)

def _nonascii(s):
    s=re.sub(r"[\W\d_]","",str(s)); return sum(ord(c)>127 for c in s)/len(s) if s else 0.0

def _jac(a,b):
    return len(a&b)/len(a|b) if (a or b) else 0.0

def _numsuffix(x,y):
    if not x or not y: return 0
    return int(x!=y and (x.endswith(y) or y.endswith(x)))

def pair_features(df):
    """df needs a_name,a_addr,a_country,b_name,b_addr,b_country,name_cos,addr_cos,in_name,in_addr."""
    out=[]
    for an,aa,ac,bn,ba,bc,ncos,acos,inn,ina in zip(df.a_name,df.a_addr,df.a_country,df.b_name,df.b_addr,df.b_country,
                                                    df.name_cos,df.addr_cos,df.in_name,df.in_addr):
        n1,n2=nn_c(an),nn_c(bn); d1,d2=na_c(aa,ac),na_c(ba,bc)
        c1,c2=n1.replace(" ",""),n2.replace(" ",""); t1,t2=set(n1.split()),set(n2.split()); u1,u2=set(d1.split()),set(d2.split())
        num1,num2=re.findall(r"\d+",str(aa)),re.findall(r"\d+",str(ba))
        w1,w2=[t for t in u1 if not t.isdigit()],[t for t in u2 if not t.isdigit()]
        l1,l2=_legal(an),_legal(bn)
        out.append(dict(
            name_cos=ncos,addr_cos=acos,in_name=inn,in_addr=ina,
            n_ratio=fuzz.ratio(n1,n2),n_tsort=fuzz.token_sort_ratio(n1,n2),n_tset=fuzz.token_set_ratio(n1,n2),n_part=fuzz.partial_ratio(n1,n2),
            n_compact=fuzz.ratio(c1,c2),n_compact_part=fuzz.partial_ratio(c1,c2),n_jac=_jac(t1,t2),
            n_contained=int(bool(c1) and bool(c2) and (c1 in c2 or c2 in c1)),n_first=int(bool(t1) and bool(t2) and n1.split()[0]==n2.split()[0]),
            n_lenratio=min(len(n1),len(n2))/max(1,max(len(n1),len(n2))),n_ntok_a=len(t1),n_ntok_b=len(t2),
            n_raw_ratio=fuzz.ratio(str(an).lower(),str(bn).lower()),
            a_ratio=fuzz.ratio(d1,d2),a_tsort=fuzz.token_sort_ratio(d1,d2),a_tset=fuzz.token_set_ratio(d1,d2),a_part=fuzz.partial_ratio(d1,d2),
            a_jac=_jac(u1,u2),a_word_jac=_jac(set(w1),set(w2)),a_ntok_a=len(u1),a_ntok_b=len(u2),
            num_jac=_jac(set(num1),set(num2)),num_first_eq=int(bool(num1) and bool(num2) and num1[0]==num2[0]),
            num_first_suffix=_numsuffix(num1[0] if num1 else "",num2[0] if num2 else ""),
            legal_eq=int(l1==l2),legal_jac=_jac(l1,l2),
            b_domain=int(bool(_DOMAIN.search(str(bn)))),a_domain=int(bool(_DOMAIN.search(str(an)))),
            b_nonascii=_nonascii(bn),a_nonascii=_nonascii(an),b_addr_nonascii=_nonascii(ba),
            same_country=int(ac==bc)))
    return pd.DataFrame(out,index=df.index)

def add_group_features(F,keys):
    """Relative features within each (query, source) candidate group. `keys` is a DataFrame of group keys aligned with F."""
    G=F.groupby([keys[c] for c in keys.columns])
    for c in ('name_cos','addr_cos','n_tset','n_compact','a_tset','a_jac'):
        F[c+'_rel']=F[c]-G[c].transform('max'); F[c+'_rank']=G[c].rank(ascending=False,method='min')
    F['n_cands']=G['name_cos'].transform('size')
    return F
