# 19_make_patches_and_csv.py
# Spot 패치 추출 + 스팟↔유전자 매칭 CSV 생성
# 원본 기능은 유지하고, 들여쓰기/가독성/CLI 인자만 정돈

import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

# 프로젝트의 CLIPDataset을 사용합니다.
from dataset import CLIPDataset


def custom_collate_fn(batch):
    """
    DataLoader가 NumPy 배열(이미지)과 문자열(barcode)을 안전하게 다루도록 하는 collate.
    - image_patch, barcode는 리스트로 유지
    - 나머지는 default_collate
    """
    keys_to_collate = [k for k in batch[0].keys() if k not in ["image_patch", "barcode"]]
    collated = {k: torch.utils.data.default_collate([d[k] for d in batch]) for k in keys_to_collate}
    collated["image_patch"] = [d["image_patch"] for d in batch]  # numpy arrays
    collated["barcode"] = [d["barcode"] for d in batch]          # strings
    return collated


def load_gene_columns(data_root: str, fallback_n: int = 3467):
    """
    features.tsv + hvg_union.npy가 있으면 실제 유전자명 사용,
    없으면 gene_0000.. 형태로 fallback.
    """
    try:
        feat_tsv = os.path.join(data_root, "data/filtered_expression_matrices/1/features.tsv")
        hvg_npy = os.path.join(data_root, "data/filtered_expression_matrices/hvg_union.npy")
        all_gene_names = pd.read_csv(feat_tsv, sep="\t", header=None)[1].values
        hvg_mask = np.load(hvg_npy)
        gene_columns = all_gene_names[hvg_mask]
        print(f"총 {len(gene_columns)}개의 유전자 이름을 불러왔습니다.")
        return list(map(str, gene_columns))
    except Exception as e:
        print(f"[WARN] 유전자 이름 관련 파일을 찾지 못했습니다({e}). 기본 이름으로 대체합니다.")
        return [f"gene_{i:04d}" for i in range(fallback_n)]


def build_dataset_info(data_root: str):
    """
    조직 슬라이스별 경로 정보를 구성합니다.
    """
    return [
        {
            "tissue_index": 1,
            "image_path": os.path.join(data_root, "image/GSM7697868_GEX_C73_A1_Merged.tiff"),
            "spatial_pos_path": os.path.join(data_root, "data/tissue_pos_matrices/tissue_positions_list_1.csv"),
            "reduced_mtx_path": os.path.join(data_root, "data/filtered_expression_matrices/1/harmony_matrix.npy"),
            "barcode_path": os.path.join(data_root, "data/filtered_expression_matrices/1/barcodes.tsv"),
        },
        {
            "tissue_index": 2,
            "image_path": os.path.join(data_root, "image/GSM7697869_GEX_C73_B1_Merged.tiff"),
            "spatial_pos_path": os.path.join(data_root, "data/tissue_pos_matrices/tissue_positions_list_2.csv"),
            "reduced_mtx_path": os.path.join(data_root, "data/filtered_expression_matrices/2/harmony_matrix.npy"),
            "barcode_path": os.path.join(data_root, "data/filtered_expression_matrices/2/barcodes.tsv"),
        },
        {
            "tissue_index": 3,
            "image_path": os.path.join(data_root, "image/GSM7697870_GEX_C73_C1_Merged.tiff"),
            "spatial_pos_path": os.path.join(data_root, "data/tissue_pos_matrices/tissue_positions_list_3.csv"),
            "reduced_mtx_path": os.path.join(data_root, "data/filtered_expression_matrices/3/harmony_matrix.npy"),
            "barcode_path": os.path.join(data_root, "data/filtered_expression_matrices/3/barcodes.tsv"),
        },
        {
            "tissue_index": 4,
            "image_path": os.path.join(data_root, "image/GSM7697871_GEX_C73_D1_Merged.tiff"),
            "spatial_pos_path": os.path.join(data_root, "data/tissue_pos_matrices/tissue_positions_list_4.csv"),
            "reduced_mtx_path": os.path.join(data_root, "data/filtered_expression_matrices/4/harmony_matrix.npy"),
            "barcode_path": os.path.join(data_root, "data/filtered_expression_matrices/4/barcodes.tsv"),
        },
    ]


def create_formatted_dataset(
    data_root: str,
    out_csv_path: str,
    out_img_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    각 tissue 슬라이스에서 패치를 추출해 PNG로 저장하고,
    spot 메타 + 유전자 발현 벡터를 합쳐 단일 CSV로 생성합니다.
    """
    os.makedirs(out_img_dir, exist_ok=True)

    # 1) 유전자 컬럼 결정
    gene_columns = load_gene_columns(data_root)

    # 2) 슬라이스 메타 구성
    dataset_info = build_dataset_info(data_root)

    # 3) 슬라이스별 처리
    all_rows = []
    global_index = 0

    for info in dataset_info:
        dataset = CLIPDataset(
            image_path=info["image_path"],
            spatial_pos_path=info["spatial_pos_path"],
            reduced_mtx_path=info["reduced_mtx_path"],
            barcode_path=info["barcode_path"],
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
        )

        pbar = tqdm(loader, desc=f"Processing Tissue {info['tissue_index']}", unit=" batch")

        for batch in pbar:
            original_patches = batch["image_patch"]                      # list[np.ndarray]
            expressions = batch["reduced_expression"].cpu().numpy()      # (B, G)
            spatial_indices = batch["spatial_indices"]                   # tensor shape (2, B)
            spatial_coords = batch["spatial_coords"]                     # tensor shape (2, B)

            for i in range(len(batch["barcode"])):
                train_id = f"TRAIN_{global_index:04d}"
                img_path = os.path.join(out_img_dir, f"{train_id}.png")

                # 패치 저장 (원본 배열을 그대로 저장)
                # 필요 시, RGB→BGR 변환이 필요할 수 있으나 원 코드를 유지합니다.
                cv2.imwrite(img_path, original_patches[i])

                # 메타 추출
                row_idx = spatial_indices[0][i].item()
                col_idx = spatial_indices[1][i].item()
                pxl_row = spatial_coords[0][i].item()
                pxl_col = spatial_coords[1][i].item()

                # CSV 행 구성
                row_data = {
                    "id": train_id,
                    "path": img_path,
                    "tissue_index": info["tissue_index"],
                    "array_row": row_idx,
                    "array_col": col_idx,
                    "pxl_row": pxl_row,
                    "pxl_col": pxl_col,
                }

                row_data.update(zip(gene_columns, expressions[i]))
                all_rows.append(row_data)
                global_index += 1

    # 4) CSV 생성
    master_df = pd.DataFrame(all_rows)

    final_columns = [
        "id",
        "path",
        "tissue_index",
        "array_row",
        "array_col",
        "pxl_row",
        "pxl_col",
    ] + list(gene_columns)

    # 열 순서 고정
    master_df = master_df[final_columns]

    # 정렬(가능하면 고정 키 기준) — 재현성/일관성
    sort_key = ["tissue_index", "array_row", "array_col", "pxl_row", "pxl_col"]
    sort_key = [c for c in sort_key if c in master_df.columns]
    if sort_key:
        master_df = master_df.sort_values(sort_key).reset_index(drop=True)

    # 저장
    master_df.to_csv(out_csv_path, index=False)

    print("\n✅ CSV 파일 및 패치 이미지 저장 완료")
    print(f" - CSV: {out_csv_path}")
    print(f" - 이미지 디렉터리: {out_img_dir}")
    print("생성된 데이터 일부:")
    print(master_df.head())

    # (선택) 간단 메타 저장
    meta = {
        "data_root": data_root,
        "out_csv": out_csv_path,
        "out_img_dir": out_img_dir,
        "num_rows": int(len(master_df)),
        "num_genes": int(len(gene_columns)),
    }
    with open(os.path.splitext(out_csv_path)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./GSE240429_data",
                    help="GSE240429 데이터 루트 디렉터리")
    ap.add_argument("--out_csv", type=str, default="dataset.csv",
                    help="출력 CSV 경로")
    ap.add_argument("--out_img_dir", type=str, default="./train",
                    help="패치 PNG 저장 디렉터리")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_formatted_dataset(
        data_root=args.data_root,
        out_csv_path=args.out_csv,
        out_img_dir=args.out_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
