# 21_train_backbones.py
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

CFG = {"IMG_SIZE":224,"EPOCHS":30,"LR":3e-4,"BS":32,"SEED":42,"NUM_WORKERS":4}

HEG=['RPL8','GATM','UBA52','RPL6','PRDX6','RPL15','RPL18A','CD81','IFITM3','PRAP1','P4HB','RPS17','ECHS1','BHMT','MT1F','RPLP2','RIDA','RPL9','TFR2','RPS8','COL18A1','ALDH1L1','RPL17','RPL28','RPS19','SERF2','C9','RPS27','RPL29','RPL39','RPS12','RPS14','MT-ND2','CFHR1','FABP1','RPL37A','CST3','CYP3A4','MT-ND3','MT-ND1','MT-CYB','APOB','MT-ND4','MT-ATP6','MT-CO3','MT-CO1','MT-CO2','CYP2E1','SAA2','ORM1']
HVG=['TSKU','GGCX','SAT1','SLC10A1','COX7B','GPT','GSTZ1','CD74','LY6E','MT-ND1','RHOB','AKR1C4','HLA-A','IGHM','TLE5','CD99','SLC38A4','CYP2B6','APOA5','MT-ND3','ATP5IF1','VIM','HSPB1','SLCO1B3','SCD','CEBPD','TMSB10','IGHA1','C1QB','CCL14','NEAT1','TMSB4X','DCN','IGFBP7','H19','S100A8','CCDC152','IGFBP1','MT-ND4L','C7','MT-ND5','MT-ND2','MALAT1','CYP3A4','IGKC','CYP1A2','HBA1','GLUL','HBA2','HBB']
MG=["HAL","CYP3A4","VWF","SOX9","KRT7","ANXA4","ACTA2","DCN"]

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
        # ST-Net 계열 특유의 강건성 향상용: 90도 회전, 좌우/상하 뒤집기, (완만한) 색감 변형, 블러
        train=A.Compose([
            A.Resize(img_size,img_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3,5), p=0.2),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0), ToTensorV2()
        ])
    else: # mpa (기본)
        train=A.Compose([
            A.Resize(img_size,img_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0), ToTensorV2()
        ])
    eval_=A.Compose([A.Resize(img_size,img_size),
                     A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0), ToTensorV2()])
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
        p=self.df.loc[i,"path"]; img=cv2.imread(p); 
        if img is None: raise FileNotFoundError(p)
        img=self.tfm(image=img)["image"] if self.tfm else img
        if self.has_labels: return img, torch.from_numpy(self.Y[i])
        return img

def make_backbone(name):
    name=name.lower()
    if name=="resnet18":
        m=models.resnet18(pretrained=True); feat=m.fc.in_features; m.fc=nn.Identity()
        head_in=feat; bb=m
    elif name=="densenet121":
        m=models.densenet121(pretrained=True); feat=m.classifier.in_features; m.classifier=nn.Identity()
        head_in=feat; bb=m
    elif name=="mobilenet_v2":
        m=models.mobilenet_v2(pretrained=True); feat=m.classifier[1].in_features; m.classifier=nn.Identity()
        head_in=feat; bb=m
    elif name=="effnet_b7":
        m=models.efficientnet_b7(pretrained=True); feat=m.classifier[1].in_features; m.classifier=nn.Identity()
        head_in=feat; bb=m
    else:
        m=models.efficientnet_b0(pretrained=True); feat=m.classifier[1].in_features; m.classifier=nn.Identity()
        head_in=feat; bb=m
    return bb, head_in

class RegHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=3000):
        super().__init__()
        self.net=nn.Sequential(nn.Dropout(0.5), nn.Linear(in_dim,hidden), nn.ReLU(inplace=True),
                               nn.Dropout(0.5), nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)

class Model(nn.Module):
    def __init__(self, backbone_name, out_dim, hidden=3000, unfreeze_from=-9):
        super().__init__()
        self.backbone, feat = make_backbone(backbone_name)
        for p in self.backbone.parameters(): p.requires_grad=False
        if hasattr(self.backbone,"features"): 
            for p in self.backbone.features[unfreeze_from:].parameters(): p.requires_grad=True
        self.head=RegHead(feat, out_dim, hidden)
    def forward(self,x):
        f=self.backbone(x); 
        if isinstance(f, (list,tuple)): f=f[0]
        return self.head(f)

def group_pcc(pred,true,gene_cols):
    idx={g:i for i,g in enumerate(gene_cols)}; out={}
    groups={"HEG":HEG,"HVG":HVG,"MG":MG}
    for name, genes in groups.items():
        cols=[idx[g] for g in genes if g in idx]; 
        if not cols: out[name]=np.nan; continue
        P=pred[:,cols]; T=true[:,cols]; corrs=[]
        for j in range(P.shape[1]):
            x,y=P[:,j],T[:,j]
            if np.std(x)==0 or np.std(y)==0: corrs.append(0.0); continue
            c=np.corrcoef(x,y)[0,1]; corrs.append(0.0 if not np.isfinite(c) else c)
        out[name]=float(np.mean(corrs)) if corrs else np.nan
    return out

def train_one_epoch(model,dl,opt,scaler,crit,dev):
    model.train(); losses=[]
    for imgs,labels in tqdm(dl,desc="Train",leave=False):
        imgs=imgs.to(dev).float(); labels=labels.to(dev)
        opt.zero_grad(set_to_none=True)
        with autocast(): pred=model(imgs); loss=crit(pred,labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def eval_epoch(model,dl,crit,dev,gene_cols):
    model.eval(); losses=[]; preds=[]; trues=[]
    for imgs,labels in tqdm(dl,desc="Valid",leave=False):
        imgs=imgs.to(dev).float(); labels=labels.to(dev)
        with autocast(): pred=model(imgs); loss=crit(pred,labels)
        losses.append(loss.item()); preds.append(pred.float().cpu().numpy()); trues.append(labels.float().cpu().numpy())
    pred=np.concatenate(preds,0); true=np.concatenate(trues,0)
    return float(np.mean(losses)), group_pcc(pred,true,gene_cols)

def main(args):
    seed_everything(CFG["SEED"]); os.makedirs(args.out_dir, exist_ok=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=pd.read_csv(args.train_csv)
    sort_key=[c for c in ["tissue_index","array_row","array_col","pxl_row","pxl_col"] if c in df.columns]
    if sort_key: df=df.sort_values(sort_key).reset_index(drop=True)
    gene_cols=[c for c in df.columns if c not in {"id","path","tissue_index","array_row","array_col","pxl_row","pxl_col","cluster","fold"} and pd.api.types.is_numeric_dtype(df[c])]
    tr_df, va_df = train_test_split(df, test_size=0.2, random_state=CFG["SEED"], shuffle=True)
    train_tfm, eval_tfm = build_aug(args.aug, CFG["IMG_SIZE"])
    tr_ds=GeneDS(tr_df,gene_cols,train_tfm,True); va_ds=GeneDS(va_df,gene_cols,eval_tfm,True)
    tr_dl=DataLoader(tr_ds,batch_size=CFG["BS"],shuffle=True,num_workers=CFG["NUM_WORKERS"],worker_init_fn=seed_worker)
    va_dl=DataLoader(va_ds,batch_size=CFG["BS"],shuffle=False,num_workers=CFG["NUM_WORKERS"],worker_init_fn=seed_worker)
    model=Model(args.backbone, out_dim=len(gene_cols), hidden=args.hidden, unfreeze_from=args.unfreeze_from).to(dev)
    opt=optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=CFG["LR"])
    sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["EPOCHS"])
    crit=nn.MSELoss().to(dev); scaler=GradScaler()
    best=float("inf"); best_state=None
    for ep in range(1, CFG["EPOCHS"]+1):
        tr=train_one_epoch(model,tr_dl,opt,scaler,crit,dev)
        va, pcc = eval_epoch(model,va_dl,crit,dev,gene_cols)
        sch.step(va)
        print(f"[{ep:02d}] Train {tr:.5f} | Val {va:.5f} | PCC HEG {pcc['HEG']:.4f} HVG {pcc['HVG']:.4f} MG {pcc['MG']:.4f}")
        if va<best: best=va; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    if best_state is not None:
        torch.save(best_state, os.path.join(args.out_dir, f"best_{args.backbone}.pth"))
    print("Done.")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="./train.csv")
    ap.add_argument("--out_dir", default="./out_backbones")
    ap.add_argument("--backbone", default="effnet_b0", choices=["effnet_b0","resnet18","densenet121","mobilenet_v2","effnet_b7"])
    ap.add_argument("--aug", default="mpa", choices=["none","mpa","stnet"])
    ap.add_argument("--hidden", type=int, default=3000)
    ap.add_argument("--unfreeze_from", type=int, default=-9, help="EffNet류의 features[unfreeze_from:]만 학습")
    args=ap.parse_args(); main(args)
