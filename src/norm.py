import re, unicodedata

SUFFIX=r"\b(private|pvt|limited|ltd|llp|l l p|llc|l l c|inc|incorporated|corp|corporation|co|company|the|and|sarl|sas|sasu|eurl|sa|sci|snc|et)\b"
_COMB=re.compile(r"[\u0300-\u036f]")

def _strip(x):
    return _COMB.sub("",unicodedata.normalize('NFKD',str(x)))

def norm_name(x):
    x=_strip(x).lower()
    x=re.sub(r"^(india|us|usa)\s{2,}","",x)                 # country prefix seen in S3
    x=re.sub(r"\(id:?\s*\d+\)|#\s*\d+|\bid:?\s*\d+\b|\s-\s\d{6,}","",x)   # id suffixes
    x=re.sub(r"\.(com|in|net|org|co\.in)\b","",x)          # domain-style names
    x=re.sub(r"[^\w ]|_"," ",x)                            # keep unicode letters (native scripts)
    x=re.sub(SUFFIX," ",x)
    return re.sub(r"\s+"," ",x).strip()

US={'AL':'alabama','AK':'alaska','AZ':'arizona','AR':'arkansas','CA':'california','CO':'colorado','CT':'connecticut','DE':'delaware','DC':'district of columbia','FL':'florida','GA':'georgia','HI':'hawaii','ID':'idaho','IL':'illinois','IN':'indiana','IA':'iowa','KS':'kansas','KY':'kentucky','LA':'louisiana','ME':'maine','MD':'maryland','MA':'massachusetts','MI':'michigan','MN':'minnesota','MS':'mississippi','MO':'missouri','MT':'montana','NE':'nebraska','NV':'nevada','NH':'new hampshire','NJ':'new jersey','NM':'new mexico','NY':'new york','NC':'north carolina','ND':'north dakota','OH':'ohio','OK':'oklahoma','OR':'oregon','PA':'pennsylvania','RI':'rhode island','SC':'south carolina','SD':'south dakota','TN':'tennessee','TX':'texas','UT':'utah','VT':'vermont','VA':'virginia','WA':'washington','WV':'west virginia','WI':'wisconsin','WY':'wyoming'}
IN={'AP':'andhra pradesh','AR':'arunachal pradesh','AS':'assam','BR':'bihar','CG':'chhattisgarh','CH':'chandigarh','DL':'delhi','GA':'goa','GJ':'gujarat','HR':'haryana','HP':'himachal pradesh','JK':'jammu and kashmir','JH':'jharkhand','KA':'karnataka','KL':'kerala','MP':'madhya pradesh','MH':'maharashtra','MN':'manipur','ML':'meghalaya','MZ':'mizoram','NL':'nagaland','OD':'odisha','OR':'odisha','PB':'punjab','RJ':'rajasthan','SK':'sikkim','TN':'tamil nadu','TS':'telangana','TR':'tripura','UP':'uttar pradesh','UK':'uttarakhand','UT':'uttarakhand','WB':'west bengal'}
STATES={'US':US,'India':IN}      # unknown countries fall back to {} (no expansion)
STREET={'st':'street','rd':'road','ave':'avenue','av':'avenue','dr':'drive','ln':'lane','blvd':'boulevard','ct':'court','pl':'place','hwy':'highway','cir':'circle','pkwy':'parkway','sq':'square','ter':'terrace','twp':'township','mt':'mount','apt':'apartment','ste':'suite','fl':'floor','flr':'floor','bd':'boulevard','bld':'boulevard','n':'north','s':'south','e':'east','w':'west'}

def norm_addr(x,country):
    x=_strip(str(x).replace('<NULL>',' ')); st=STATES.get(country,{})
    toks=[]
    for t in re.split(r"[^\w]+",x):                         # abbreviation expansion needs original case
        if not t: continue
        if len(t)==2 and t.isupper() and t in st: toks.extend(st[t].split()); continue
        t=t.lower(); toks.append(STREET.get(t,t))
    return " ".join(toks)
