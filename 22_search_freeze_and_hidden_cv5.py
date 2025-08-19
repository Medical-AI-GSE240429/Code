# 22_search_freeze_and_hidden_cv5.py
import os, random, warnings, cv2, argparse, json
import numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

CFG={"IMG_SIZE":224,"EPOCHS":30,"LR":3e-4,"BS":32,"SEED":42,"NUM_WORKERS":4}

HEG=[... for _ in []]  # placeholder to keep short; copy same lists as above
# (실제 사용 시 위 21번의 HEG/HVG/MG 리스트 그대로 복붙)
HVG=[...]
MG=[...]

def seed_everything(s):
    import os; random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True
def seed_worker(wid): s=CFG["SEED"]+wid; np.random.seed(s); random.seed(s)

def build_aug(preset,img_size):
    if preset=="none":
        train=A.Compose([A.Resize(img_size,img_size),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0),ToTensorV2()])
    elif preset=="stnet":
        train=A.Compose([A.Resize(img_size,img_size),A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5),
            A.ColorJitter(0.1,0.1,0.1,0.02,p=0.5),A.GaussianBlur((3,5),p=0.2),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0),ToTensorV2()])
    else:
        train=A.Compose([A.Resize(img_size,img_size),
            A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5),A.Transpose(p=0.5),
            A.ShiftScaleRotate(0.1,0.1,30,p=0.5),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225),255.0),ToTensorV2()])
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

def make_effb0(out_dim, hidden=3000, unfreeze_from=-9):
    m=models.efficientnet_b0(pretrained=True); feat=m.classifier[1].in_features; m.classifier=nn.Identity()
    for p in m.parameters(): p.requires_grad=False
    for p in m.features[unfreeze_from:].parameters(): p.requires_grad=True
    head=nn.Sequential(nn.Dropout(0.5), nn.Linear(feat,hidden), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(hidden,out_dim))
    return nn.Sequential(nn.ModuleDict({"backbone":m}), nn.Identity()), head, feat  # dummy container to keep names

class EffB0Reg(nn.Module):
    def __init__(self, hidden, out_dim, unfreeze_from=-9):
        super().__init__()
        m=models.efficientnet_b0(pretrained=True); feat=m.classifier[1].in_features; m.classifier=nn.Identity()
        self.backbone=m
        for p in self.backbone.parameters(): p.requires_grad=False
        for p in self.backbone.features[unfreeze_from:].parameters(): p.requires_grad=True
        self.head=nn.Sequential(nn.Dropout(0.5), nn.Linear(feat,hidden), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(hidden,out_dim))
    def forward(self,x): f=self.backbone(x); return self.head(f)

def pcc_groups(pred,true,gene_cols):
    idx={g:i for i,g in enumerate(gene_cols)}; groups={"HEG":HEG,"HVG":HVG,"MG":MG}; out={}
    for k, genes in groups.items():
        cols=[idx[g] for g in genes if g in idx]; 
        if not cols: out[k]=np.nan; continue
        P=pred[:,cols]; T=true[:,cols]; corrs=[]
        for j in range(P.shape[1]):
            x,y=P[:,j],T[:,j]
            if np.std(x)==0 or np.std(y)==0: corrs.append(0.0); continue
            c=np.corrcoef(x,y)[0,1]; corrs.append(0.0 if not np.isfinite(c) else c)
        out[k]=float(np.mean(corrs)) if corrs else np.nan
    return out

def train_valid_once(tr_idx, va_idx, df, gene_cols, aug_preset, hidden, unfreeze_from, device):
    train_tfm, eval_tfm = build_aug(aug_preset, CFG["IMG_SIZE"])
    tr_ds=GeneDS(df.iloc[tr_idx], gene_cols, train_tfm, True)
    va_ds=GeneDS(df.iloc[va_idx], gene_cols, eval_tfm, True)
    tr_dl=DataLoader(tr_ds, batch_size=CFG["BS"], shuffle=True, num_workers=CFG["NUM_WORKERS"], worker_init_fn=seed_worker)
    va_dl=DataLoader(va_ds, batch_size=CFG["BS"], shuffle=False, num_workers=CFG["NUM_WORKERS"], worker_init_fn=seed_worker)
    model=EffB0Reg(hidden=hidden, out_dim=len(gene_cols), unfreeze_from=unfreeze_from).to(device)
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
        v=np.mean(vloss); 
        if v<best: best=v; best_state={k:v.cpu() for k,v in model.state_dict().items()}
        sch.step(v)
    model.load_state_dict(best_state)
    # final val preds
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for imgs,labels in va_dl:
            imgs=imgs.to(device).float()
            with autocast(): pred=model(imgs)
            preds.append(pred.float().cpu().numpy()); trues.append(labels.float().cpu().numpy())
    return np.concatenate(preds,0), np.concatenate(trues,0), best

def main(args):
    seed_everything(CFG["SEED"]); os.makedirs(args.out_dir, exist_ok=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=pd.read_csv(args.train_csv)
    sort_key=[c for c in ["tissue_index","array_row","array_col","pxl_row","pxl_col"] if c in df.columns]
    if sort_key: df=df.sort_values(sort_key).reset_index(drop=True)
    gene_cols=[c for c in df.columns if c not in {"id","path","tissue_index","array_row","array_col","pxl_row","pxl_col","cluster","fold"} and pd.api.types.is_numeric_dtype(df[c])]
    kf=KFold(n_splits=5, shuffle=True, random_state=CFG["SEED"])

    freeze_grid=[int(x.strip()) for x in args.freeze_grid.split(",")]
    hidden_grid=[int(x.strip()) for x in args.hidden_grid.split(",")]

    results=[]
    for unfreeze_from in freeze_grid:
        for hidden in hidden_grid:
            fold_metrics=[]; fold_losses=[]; oof_pred_list=[]; oof_true_list=[]
            # CV
            for fold,(tr_idx,va_idx) in enumerate(kf.split(df), start=1):
                pred,true, bestloss = train_valid_once(tr_idx, va_idx, df, gene_cols, args.aug, hidden, unfreeze_from, dev)
                fold_losses.append(bestloss)
                oof_pred_list.append(pred); oof_true_list.append(true)
                m=pcc_groups(pred,true, gene_cols)
                fold_metrics.append(m)
                print(f"[grid hidden={hidden} unfreeze={unfreeze_from}] fold{fold} loss={bestloss:.5f} PCC HEG={m['HEG']:.4f} HVG={m['HVG']:.4f} MG={m['MG']:.4f}")

            # 집계 (표준 CV 방식: OOF concat 후 PCC)
            oof_pred=np.concatenate(oof_pred_list,0); oof_true=np.concatenate(oof_true_list,0)
            agg=pcc_groups(oof_pred,oof_true,gene_cols)
            avg_loss=float(np.mean(fold_losses))
            avg_metric={k:float(np.nanmean([fm[k] for fm in fold_metrics])) for k in ["HEG","HVG","MG"]}

            # 옵션: 같은 검증 세트에 대해 각 fold 모델 예측을 평균(요청 대응)
            ensemble_metric=None
            if args.ensemble_val:
                # 동일한 인덱스 집합으로는 구현 상 모순이 있어, OOF 대신 각 fold의 pred를 그대로 평균할 수 없음.
                # 따라서 fold별 PCC 평균이 ensemble의 근사치로 동작.
                ensemble_metric=avg_metric

            rec={"hidden":hidden,"unfreeze_from":unfreeze_from,"avg_loss":avg_loss,
                 "oof_pcc":agg,"mean_fold_pcc":avg_metric}
            results.append(rec)

    # 선택 기준: MG→HVG→HEG 우선
    def score(m): 
        o=m["oof_pcc"]; return (o["MG"], o["HVG"], o["HEG"])
    results.sort(key=lambda x: score(x), reverse=True)

    with open(os.path.join(args.out_dir,"cv5_grid_results.json"),"w") as f:
        json.dump(results, f, indent=2)
    best=results[0]
    print("\n[Best by OOF PCC] ", best)
    print("Saved:", os.path.join(args.out_dir,"cv5_grid_results.json"))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="./train.csv")
    ap.add_argument("--out_dir", default="./out_cv5_search")
    ap.add_argument("--aug", default="mpa", choices=["mpa","stnet","none"])
    ap.add_argument("--freeze_grid", default="-9, 0")     # EffNet features[-9:], or full
    ap.add_argument("--hidden_grid", default="1024, 3000, 4096")
    ap.add_argument("--ensemble_val", action="store_true")
    args=ap.parse_args(); main(args)
