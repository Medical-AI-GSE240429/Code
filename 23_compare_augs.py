# 23_compare_augs.py
import os, random, warnings, cv2, argparse
import numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

CFG={"IMG_SIZE":224,"EPOCHS":30,"LR":3e-4,"BS":32,"SEED":42,"NUM_WORKERS":4}

HEG=[...]; HVG=[...]; MG=[...]  # 21번과 동일 리스트 복붙

def seed_everything(s):
    import os; random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True
def seed_worker(wid): s=CFG["SEED"]+wid; np.random.seed(s); random.seed(s)

def build_aug(preset, img_size):
    if preset=="none":
        train=A.Compose([A.Resize(img_size,img_size),
                         A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0),ToTensorV2()])
    elif preset=="stnet":
        train=A.Compose([
            A.Resize(img_size,img_size), A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.ColorJitter(0.1,0.1,0.1,0.02,p=0.5),
            A.GaussianBlur((3,5),p=0.2),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0), ToTensorV2()
        ])
    else:
        train=A.Compose([
            A.Resize(img_size,img_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Transpose(p=0.5),
            A.ShiftScaleRotate(0.1,0.1,30,p=0.5),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0), ToTensorV2()
        ])
    eval_=A.Compose([A.Resize(img_size,img_size),
                     A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0),ToTensorV2()])
    return train, eval_

def detect_gene_cols(df):
    fixed={"id","path","tissue_index","array_row","array_col","pxl_row","pxl_col","cluster","fold"}
    return [c for c in df.columns if c not in fixed and pd.api.types.is_numeric_dtype(df[c])]

class GeneDS(Dataset):
    def __init__(self, df, gene_cols, tfm, has_labels=True):
        self.df=df.reset_index(drop=True); self.gene_cols=gene_cols; self.tfm=tfm
        self.has_labels=has_labels and all(c in df.columns for c in gene_cols)
        self.Y=None if not self.has_labels else df[gene_cols].to_numpy(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        p=self.df.loc[i,"path"]; img=cv2.imread(p)
        if img is None: raise FileNotFoundError(p)
        img=self.tfm(image=img)["image"] if self.tfm else img
        if self.has_labels: return img, torch.from_numpy(self.Y[i])
        return img

class EffB0Reg(nn.Module):
    def __init__(self, out_dim, hidden=3000, unfreeze_from=-9):
        super().__init__()
        m=models.efficientnet_b0(pretrained=True); feat=m.classifier[1].in_features; m.classifier=nn.Identity()
        self.backbone=m
        for p in self.backbone.parameters(): p.requires_grad=False
        for p in self.backbone.features[unfreeze_from:].parameters(): p.requires_grad=True
        self.head=nn.Sequential(nn.Dropout(0.5), nn.Linear(feat,hidden), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(hidden,out_dim))
    def forward(self,x): f=self.backbone(x); return self.head(f)

def group_pcc(pred,true,gene_cols):
    idx={g:i for i,g in enumerate(gene_cols)}; out={}
    for name,genes in {"HEG":HEG,"HVG":HVG,"MG":MG}.items():
        cols=[idx[g] for g in genes if g in idx]; 
        if not cols: out[name]=np.nan; continue
        P=pred[:,cols]; T=true[:,cols]; corrs=[]
        for j in range(P.shape[1]):
            x,y=P[:,j],T[:,j]
            if np.std(x)==0 or np.std(y)==0: corrs.append(0.0); continue
            c=np.corrcoef(x,y)[0,1]; corrs.append(0.0 if not np.isfinite(c) else c)
        out[name]=float(np.mean(corrs)) if corrs else np.nan
    return out

def train_eval_once(df, gene_cols, aug_preset, device, out_dim, hidden, unfreeze_from):
    tr,va = train_test_split(df, test_size=0.2, random_state=CFG["SEED"], shuffle=True)
    tr_tfm, ev_tfm = build_aug(aug_preset, CFG["IMG_SIZE"])
    tr_ds=GeneDS(tr,gene_cols,tr_tfm,True); va_ds=GeneDS(va,gene_cols,ev_tfm,True)
    tr_dl=DataLoader(tr_ds,batch_size=CFG["BS"],shuffle=True,num_workers=CFG["NUM_WORKERS"],worker_init_fn=seed_worker)
    va_dl=DataLoader(va_ds,batch_size=CFG["BS"],shuffle=False,num_workers=CFG["NUM_WORKERS"],worker_init_fn=seed_worker)
    model=EffB0Reg(out_dim, hidden, unfreeze_from).to(device)
    opt=optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=CFG["LR"])
    sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["EPOCHS"])
    crit=nn.MSELoss().to(device); scaler=GradScaler()
    best=float("inf"); best_state=None
    for ep in range(1, CFG["EPOCHS"]+1):
        # train
        model.train(); losses=[]
        for imgs,labels in tr_dl:
            imgs=imgs.to(device).float(); labels=labels.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(): pred=model(imgs); loss=crit(pred,labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        # valid
        model.eval(); vloss=[]; preds=[]; trues=[]
        with torch.no_grad():
            for imgs,labels in va_dl:
                imgs=imgs.to(device).float(); labels=labels.to(device)
                with autocast(): pred=model(imgs); loss=crit(pred,labels)
                vloss.append(loss.item()); preds.append(pred.float().cpu().numpy()); trues.append(labels.float().cpu().numpy())
        v=np.mean(vloss); sch.step(v)
        if v<best: best=v; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred=np.concatenate(preds,0); true=np.concatenate(trues,0)
    return best, group_pcc(pred,true,gene_cols)

def main(args):
    seed_everything(CFG["SEED"]); os.makedirs(args.out_dir, exist_ok=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=pd.read_csv(args.train_csv)
    sort_key=[c for c in ["tissue_index","array_row","array_col","pxl_row","pxl_col"] if c in df.columns]
    if sort_key: df=df.sort_values(sort_key).reset_index(drop=True)
    gene_cols=detect_gene_cols(df)
    augs=[a.strip() for a in args.augs.split(",")]
    rows=[]
    for aug in augs:
        val_loss, pcc = train_eval_once(df, gene_cols, aug, dev, len(gene_cols), args.hidden, args.unfreeze_from)
        rows.append({"aug":aug, "val_loss":val_loss, "HEG":pcc["HEG"], "HVG":pcc["HVG"], "MG":pcc["MG"]})
        print(f"[{aug}] ValLoss {val_loss:.5f} | PCC HEG {pcc['HEG']:.4f} HVG {pcc['HVG']:.4f} MG {pcc['MG']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir,"aug_compare.csv"), index=False)
    print("Saved:", os.path.join(args.out_dir,"aug_compare.csv"))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="./train.csv")
    ap.add_argument("--out_dir", default="./out_aug_compare")
    ap.add_argument("--augs", default="mpa, stnet, none")
    ap.add_argument("--hidden", type=int, default=3000)
    ap.add_argument("--unfreeze_from", type=int, default=-9)
    args=ap.parse_args(); main(args)
