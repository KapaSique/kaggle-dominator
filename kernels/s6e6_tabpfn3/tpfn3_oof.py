"""S6E6 — TabPFN-v3 OOF, done RIGHT (per philippsinger recipe).
Our old TabPFN was cloud-client CTX=3000 (crippled -> 0.93). This uses the LOCAL
TabPFN-3 weights (gated Kaggle model prior-labsai/tabpfn-3), balance_probabilities
=True (for balanced_accuracy), large context. Honest 5-fold OOF + test proba.

Model delivery: the model_sources push mechanism is broken on the current Kaggle
kernels API ("Notebook not found"). License is ACCEPTED on the account, so we pull
the gated weights at RUNTIME via kagglehub (internet on) and point
TABPFN_MODEL_CACHE_DIR at the downloaded dir (== philippsinger's mounted path)."""
import os, glob, time, warnings, subprocess, sys; warnings.filterwarnings("ignore")
t0=time.time()
def log(s): print(f"[{time.time()-t0:.0f}s] {s}",flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","tabpfn","kagglehub"],check=False,timeout=900)

# --- get the gated TabPFN-3 weights ---
# primary: kagglehub runtime download (license accepted + internet). fallback:
# model_sources mount if a future push ever attaches it.
MODELDIR=None
try:
    import kagglehub
    MODELDIR=kagglehub.model_download("prior-labsai/tabpfn-3/pytorch/default/1")
    log(f"kagglehub model dir: {MODELDIR}")
except Exception as e:
    log(f"kagglehub download FAILED: {e}")
if not MODELDIR or not glob.glob(os.path.join(MODELDIR,"*.ckpt")):
    for cand in glob.glob("/kaggle/input/**/tabpfn-3/**/", recursive=True):
        if glob.glob(os.path.join(cand,"*.ckpt")): MODELDIR=cand; break
assert MODELDIR and glob.glob(os.path.join(MODELDIR,"*.ckpt")), f"no .ckpt found (dir={MODELDIR})"
os.environ["TABPFN_MODEL_CACHE_DIR"]=MODELDIR
log(f"TABPFN_MODEL_CACHE_DIR={MODELDIR}  ckpts={[os.path.basename(p) for p in glob.glob(os.path.join(MODELDIR,'*.ckpt'))][:6]}")

import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
def find(p):
    g=glob.glob(f"/kaggle/input/**/{p}",recursive=True); return g[0] if g else None
classes=["GALAXY","QSO","STAR"]
tr=pd.read_csv(find("train.csv")); te=pd.read_csv(find("test.csv"))
y=tr["class"].map({c:i for i,c in enumerate(classes)}).values; N=len(tr); M=len(te)
def fe(df):
    df=df.copy(); b=["u","g","r","i","z"]
    for a,c in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("u","z"),("u","i"),("g","z")]:
        df[f"{a}_{c}"]=df[a]-df[c]
    df["redshift_log"]=np.log1p(df["redshift"].clip(lower=0))
    for c in ["spectral_type","galaxy_population"]:
        if c in df.columns: df[c]=df[c].astype("category").cat.codes
    return df
trX=fe(tr); teX=fe(te); feats=[c for c in trX.columns if c not in ["id","class"]]
Xtr=trX[feats].values.astype(np.float32); Xte=teX[feats].values.astype(np.float32)
from tabpfn import TabPFNClassifier
def mk():
    return TabPFNClassifier(device="cuda", n_estimators=2, balance_probabilities=True,
                            ignore_pretraining_limits=True)
CTX=50000  # v3 local handles big context (vs our old crippled 3000)
skf=StratifiedKFold(5,shuffle=True,random_state=42)
oof=np.zeros((N,3)); pred=np.zeros((M,3))
for f,(trn,va) in enumerate(skf.split(Xtr,y)):
    rng=np.random.RandomState(f)
    sub = trn if len(trn)<=CTX else rng.choice(trn,CTX,replace=False)
    clf=mk(); clf.fit(Xtr[sub],y[sub])
    oof[va]=clf.predict_proba(Xtr[va])
    pred+=clf.predict_proba(Xte)/5
    log(f"fold{f} BA={balanced_accuracy_score(y[va],oof[va].argmax(1)):.5f}")
ba=balanced_accuracy_score(y,oof.argmax(1))
log(f">>> TabPFN-3 OOF BA = {ba:.5f}")
np.save("/kaggle/working/oof_tabpfn3.npy",oof.astype("float32"))
np.save("/kaggle/working/test_tabpfn3.npy",pred.astype("float32"))
open("/kaggle/working/scores.txt","w").write(f"tabpfn3_oof_ba={ba:.5f}\n")
log("DONE")
