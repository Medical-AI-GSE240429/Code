# 20_train_effb0_sota_split8020.py
import os, random, warnings, cv2, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

CFG = {
    "IMG_SIZE": 224,
    "EPOCHS": 30,
    "LEARNING_RATE": 3e-4,
    "BATCH_SIZE": 32,
    "SEED": 42,
    "NUM_WORKERS": 4,
}

HEG = ['RPL8','GATM','UBA52','RPL6','PRDX6','RPL15','RPL18A','CD81','IFITM3','PRAP1','P4HB','RPS17',
       'ECHS1','BHMT','MT1F','RPLP2','RIDA','RPL9','TFR2','RPS8','COL18A1','ALDH1L1','RPL17','RPL28',
       'RPS19','SERF2','C9','RPS27','RPL29','RPL39','RPS12','RPS14','MT-ND2','CFHR1','FABP1','RPL37A',
       'CST3','CYP3A4','MT-ND3','MT-ND1','MT-CYB','APOB','MT-ND4','MT-ATP6','MT-CO3','MT-CO1','MT-CO2',
       'CYP2E1','SAA2','ORM1']
HVG = ['TSKU','GGCX','SAT1','SLC10A1','COX7B','GPT','GSTZ1','CD74','LY6E','MT-ND1','RHOB','AKR1C4',
       'HLA-A','IGHM','TLE5','CD99','SLC38A4','CYP2B6','APOA5','MT-ND3','ATP5IF1','VIM','HSPB1',
       'SLCO1B3','SCD','CEBPD','TMSB10','IGHA1','C1QB','CCL14','NEAT1','TMSB4X','DCN','IGFBP7','H19',
       'S100A8','CCDC152','IGFBP1','MT-ND4L','C7','MT-ND5','MT-ND2','MALAT1','CYP3A4','IGKC','CYP1A2',
       'HBA1','GLUL','HBA2','HBB']
MG  = ["HAL","CYP3A4","VWF","SOX9","KRT7","ANXA4","ACTA2","DCN"]

def seed_everything(seed: int):
    import os
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True

def seed_worker(worker_id):
    s = CFG["SEED"] + worker_id
    np.random.seed(s); random.seed(s)

def build_transforms(img_size):
    train_tfm = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    eval_tfm = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return train_tfm, eval_tfm

def detect_gene_cols(df: pd.DataFrame):
    fixed = {"id","path","tissue_index","array_row","array_col","pxl_row","pxl_col","cluster","fold"}
    gene_cols = [c for c in df.columns if c not in fixed and pd.api.types.is_numeric_dtype(df[c])]
    if not gene_cols:
        raise ValueError("유전자 열을 찾지 못했습니다. CSV 스키마를 확인하세요.")
    return gene_cols

def split_train_valid_8020(df: pd.DataFrame, seed: int):
    """폴드 없이 dataset.csv를 8:2로 고정 분할"""
    tr_df, va_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)

class GeneDataset(Dataset):
    def __init__(self, df, gene_cols, tfm=None, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.gene_cols = gene_cols
        self.tfm = tfm
        self.has_labels = has_labels and all(c in df.columns for c in gene_cols)
        self.Y = None if not self.has_labels else df[gene_cols].to_numpy(np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        path = self.df.loc[i, "path"]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"이미지 로드 실패: {path}")
        if self.tfm is not None:
            img = self.tfm(image=img)["image"]
        if self.has_labels:
            return img, torch.from_numpy(self.Y[i])
        return img

class EffB0Reg(nn.Module):
    def __init__(self, out_dim, hidden_size=3000):
        super().__init__()
        m = models.efficientnet_b0(pretrained=True)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        self.backbone = m
        self.dropout = nn.Dropout(0.5)
        # 기본: 대부분 동결, 마지막 블록들만 학습
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone.features[-9:].parameters(): p.requires_grad = True
        self.hidden = nn.Linear(feat_dim, hidden_size)
        self.act = nn.ReLU(inplace=True)
        self.regressor = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        f = self.backbone(x)
        z = self.dropout(f)
        z = self.hidden(z); z = self.act(z)
        z = self.dropout(z)
        y = self.regressor(z)
        return y

def train_one_epoch(model, dl, optimizer, scaler, criterion, device):
    model.train(); losses=[]
    for imgs, labels in tqdm(dl, desc="Train", leave=False):
        imgs = imgs.to(device).float(); labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            pred = model(imgs)
            loss = criterion(pred, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def eval_epoch(model, dl, criterion, device, gene_cols):
    model.eval(); losses=[]; preds=[]; trues=[]
    for imgs, labels in tqdm(dl, desc="Valid", leave=False):
        imgs = imgs.to(device).float(); labels = labels.to(device)
        with autocast():
            pred = model(imgs)
            loss = criterion(pred, labels)
        losses.append(loss.item())
        preds.append(pred.float().cpu().numpy()); trues.append(labels.float().cpu().numpy())
    pred = np.concatenate(preds, 0); true = np.concatenate(trues, 0)
    pcc = group_pcc(pred, true, gene_cols)
    return float(np.mean(losses)), pcc

def group_pcc(pred, true, gene_cols):
    idx = {g:i for i,g in enumerate(gene_cols)}
    out={}
    for name, genes in {"HEG":HEG,"HVG":HVG,"MG":MG}.items():
        cols = [idx[g] for g in genes if g in idx]
        if not cols: out[name]=np.nan; continue
        P = pred[:, cols]; T = true[:, cols]
        corrs=[]
        for j in range(P.shape[1]):
            x, y = P[:, j], T[:, j]
            if np.std(x)==0 or np.std(y)==0: corrs.append(0.0); continue
            c = np.corrcoef(x, y)[0,1]
            corrs.append(0.0 if not np.isfinite(c) else c)
        out[name] = float(np.mean(corrs)) if corrs else np.nan
    return out

@torch.no_grad()
def infer_epoch(model, dl, device):
    model.eval(); preds=[]
    for batch in tqdm(dl, desc="Infer", leave=False):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device).float()
        with autocast():
            pred = model(imgs)
        preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds, 0)

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(CFG["SEED"])
    os.makedirs(args.out_dir, exist_ok=True)

    # CSV 로드 및 정렬(가능하면 고정 키)
    df = pd.read_csv(args.train_csv)
    sort_key = [c for c in ["tissue_index","array_row","array_col","pxl_row","pxl_col"] if c in df.columns]
    if sort_key: df = df.sort_values(sort_key).reset_index(drop=True)
    gene_cols = detect_gene_cols(df)

    # === 8:2 분할 ===
    tr_df, va_df = split_train_valid_8020(df, seed=CFG["SEED"])

    # 테스트 CSV
    test_df = pd.read_csv(args.test_csv)
    if sort_key: test_df = test_df.sort_values(sort_key).reset_index(drop=True)
    test_has_labels = all(c in test_df.columns for c in gene_cols)

    # 변환/로더
    train_tfm, eval_tfm = build_transforms(CFG["IMG_SIZE"])
    tr_ds = GeneDataset(tr_df, gene_cols, tfm=train_tfm, has_labels=True)
    va_ds = GeneDataset(va_df, gene_cols, tfm=eval_tfm,   has_labels=True)
    tr_dl = DataLoader(tr_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True,
                       num_workers=CFG["NUM_WORKERS"], worker_init_fn=seed_worker)
    va_dl = DataLoader(va_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False,
                       num_workers=CFG["NUM_WORKERS"], worker_init_fn=seed_worker)

    # 모델/학습
    model = EffB0Reg(out_dim=len(gene_cols)).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["EPOCHS"])
    criterion = nn.MSELoss().to(device)
    scaler = GradScaler()

    best_loss = float("inf"); best_state=None
    for epoch in range(1, CFG["EPOCHS"]+1):
        tr_loss = train_one_epoch(model, tr_dl, optimizer, scaler, criterion, device)
        va_loss, va_pcc = eval_epoch(model, va_dl, criterion, device, gene_cols)
        if scheduler is not None: scheduler.step(va_loss)
        print(f"[{epoch:02d}] Train={tr_loss:.5f} | Val={va_loss:.5f} | "
              f"PCC HEG={va_pcc['HEG']:.4f} HVG={va_pcc['HVG']:.4f} MG={va_pcc['MG']:.4f}")
        if va_loss < best_loss:
            best_loss=va_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.out_dir, "best_effb0.pth"))
    if best_state is not None:
        model.load_state_dict(best_state)

    # 테스트
    if test_has_labels:
        te_ds = GeneDataset(test_df, gene_cols, tfm=eval_tfm, has_labels=True)
        te_dl = DataLoader(te_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=CFG["NUM_WORKERS"])
        te_loss, te_pcc = eval_epoch(model, te_dl, criterion, device, gene_cols)
        print(f"[TEST] Loss={te_loss:.5f} | PCC HEG={te_pcc['HEG']:.4f} HVG={te_pcc['HVG']:.4f} MG={te_pcc['MG']:.4f}")
    else:
        te_ds = GeneDataset(test_df, gene_cols, tfm=eval_tfm, has_labels=False)
        te_dl = DataLoader(te_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=CFG["NUM_WORKERS"])
        te_pred = infer_epoch(model, te_dl, device)
        out = pd.DataFrame(te_pred, columns=gene_cols)
        if "id" in test_df.columns: out.insert(0, "id", test_df["id"].values)
        out.to_csv(os.path.join(args.out_dir, "pred_test.csv"), index=False)
        print("Saved:", os.path.join(args.out_dir, "pred_test.csv"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="./train.csv")
    ap.add_argument("--test_csv",  type=str, default="./test.csv")
    ap.add_argument("--out_dir",   type=str, default="./out_effb0_split8020")
    args = ap.parse_args()
    main(args)
