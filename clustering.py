# ============================================================================
# 클러스터링 결과 시각화 및 샘플 저장 코드
# ============================================================================

import os
import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter

def create_cluster_visualization_and_samples():
    """
    가중치 (2.0, 500) 조합으로 클러스터링 수행 후 시각화 및 샘플 저장
    """
    print(" Starting cluster visualization with weights (2.0, 500)...")
    
    # 1. 기존 데이터 로드
    df = pd.read_csv('dataset.csv')
    df = df[df['tissue_index'].isin([1, 2, 4])]
    print(f"Using {len(df)} samples from tissues 1, 2, 4")
    
    # 2. 기존 특징들 로드 (이미 추출된 것으로 가정)
    # 여기서는 main() 함수에서 추출한 특징들을 사용한다고 가정
    # 실제로는 이전 단계에서 저장된 특징을 로드하거나 다시 추출
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TissueDataset('train', df, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 특징 추출
    print(" Extracting features...")
    pos_features = create_positional_features(df, pos_dim=64)
    extractor = HistologyFeatureExtractor(model_path='weights/tenpercent_resnet18.ckpt')
    resnet_features, names = extractor.extract_features(dataloader)
    
    # 3. 특정 가중치로 클러스터링
    print(" Clustering with weights (2.0, 500)...")
    fused_features = fuse_features(resnet_features, pos_features,
                                 resnet_weight=2.0, pos_weight=500.0)
    
    umap_embedded, clusters, metrics = perform_clustering_with_eval(fused_features)
    
    # 4. 결과 정리
    results_df = pd.DataFrame({
        'image_name': names,
        'cluster': clusters,
        'array_row': df['array_row'].values,
        'array_col': df['array_col'].values,
        'tissue_index': df['tissue_index'].values,
        'umap_x': umap_embedded[:, 0],
        'umap_y': umap_embedded[:, 1]
    })
    
    # 5. 결과 폴더 생성
    result_dir = 'clustering_results_2.0_500'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, 'cluster_samples'))
    
    print(f" Clustering Results:")
    print(f"   - Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"   - Number of clusters: {metrics['n_clusters']}")
    
    # 클러스터 분포 확인
    cluster_counts = Counter(clusters)
    print("\ Cluster Distribution:")
    for cluster_id in sorted(cluster_counts.keys()):
        if cluster_id == -1:
            print(f"   Noise: {cluster_counts[cluster_id]} samples")
        else:
            print(f"   Cluster {cluster_id}: {cluster_counts[cluster_id]} samples")
    
    # 6. 전체 UMAP 시각화
    create_overall_visualization(results_df, result_dir)
    
    # 7. 각 클러스터별 샘플 저장 및 시각화
    save_cluster_samples(results_df, result_dir, n_samples=50)
    
    # 8. 조직별 분포 분석
    analyze_tissue_distribution(results_df, result_dir)
    
    # 9. 결과 CSV 저장
    results_df.to_csv(os.path.join(result_dir, 'clustering_results.csv'), index=False)
    
    print(f"\n Complete! Results saved to '{result_dir}' folder")

def create_overall_visualization(results_df, result_dir):
    """전체 클러스터링 결과 시각화"""
    print(" Creating overall UMAP visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 클러스터별 색상
    noise_mask = results_df['cluster'] == -1
    cluster_mask = ~noise_mask
    
    # 노이즈 포인트
    if np.any(noise_mask):
        axes[0].scatter(results_df[noise_mask]['umap_x'], results_df[noise_mask]['umap_y'],
                       c='lightgray', alpha=0.5, s=15, label='Noise')
    
    # 클러스터 포인트
    if np.any(cluster_mask):
        scatter = axes[0].scatter(results_df[cluster_mask]['umap_x'], results_df[cluster_mask]['umap_y'],
                                 c=results_df[cluster_mask]['cluster'], cmap='tab20', 
                                 alpha=0.8, s=20)
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    axes[0].set_title('Clustering Results (Weights: 2.0, 500)', fontsize=14)
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. 조직별 색상
    tissue_colors = {1: 'red', 2: 'blue', 4: 'green'}
    for tissue_id in [1, 2, 4]:
        tissue_mask = results_df['tissue_index'] == tissue_id
        axes[1].scatter(results_df[tissue_mask]['umap_x'], results_df[tissue_mask]['umap_y'],
                       c=tissue_colors[tissue_id], alpha=0.7, s=20, 
                       label=f'Tissue {tissue_id}')
    
    axes[1].set_title('Tissue Distribution', fontsize=14)
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. 클러스터 크기 히스토그램
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    cluster_ids = []
    counts = []
    
    for cluster_id, count in cluster_counts.items():
        if cluster_id != -1:  # 노이즈 제외
            cluster_ids.append(f'C{cluster_id}')
            counts.append(count)
    
    if cluster_ids:
        axes[2].bar(cluster_ids, counts, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2].set_title('Cluster Sizes', fontsize=14)
        axes[2].set_xlabel('Cluster')
        axes[2].set_ylabel('Number of Samples')
        axes[2].tick_params(axis='x', rotation=45)
    
    # 노이즈 포인트도 표시
    if -1 in cluster_counts:
        axes[2].bar(['Noise'], [cluster_counts[-1]], alpha=0.7, color='lightgray', edgecolor='black')
    
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'overall_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_cluster_samples(results_df, result_dir, n_samples=50):
    """각 클러스터별 샘플 이미지 저장 및 미리보기 생성"""
    print(f" Saving {n_samples} samples per cluster...")
    
    unique_clusters = sorted([c for c in results_df['cluster'].unique() if c != -1])
    
    for cluster_id in unique_clusters:
        cluster_data = results_df[results_df['cluster'] == cluster_id]
        print(f"   Processing Cluster {cluster_id}: {len(cluster_data)} total samples")
        
        # 클러스터 폴더 생성
        cluster_dir = os.path.join(result_dir, 'cluster_samples', f'cluster_{cluster_id}')
        os.makedirs(cluster_dir)
        
        # 샘플 선택 (랜덤하게 n_samples개 또는 전체)
        sample_size = min(n_samples, len(cluster_data))
        sampled_data = cluster_data.sample(n=sample_size, random_state=42)
        
        # 개별 이미지 저장
        saved_images = []
        for idx, row in sampled_data.iterrows():
            img_name = row['image_name']
            src_path = os.path.join('train', img_name)
            dst_path = os.path.join(cluster_dir, img_name)
            
            try:
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    saved_images.append(img_name)
                else:
                    print(f"     Warning: Image {img_name} not found")
            except Exception as e:
                print(f"     Error copying {img_name}: {e}")
        
        print(f"     Saved {len(saved_images)} images to cluster_{cluster_id}/")
        
        # 클러스터 미리보기 그리드 생성 (5x10 = 50개)
        create_cluster_preview_grid(cluster_dir, saved_images[:50], cluster_id, result_dir)
        
        # 클러스터 통계 저장
        cluster_stats = {
            'cluster_id': cluster_id,
            'total_samples': len(cluster_data),
            'saved_samples': len(saved_images),
            'tissue_distribution': cluster_data['tissue_index'].value_counts().to_dict(),
            'avg_umap_x': cluster_data['umap_x'].mean(),
            'avg_umap_y': cluster_data['umap_y'].mean(),
            'std_umap_x': cluster_data['umap_x'].std(),
            'std_umap_y': cluster_data['umap_y'].std()
        }
        
        # JSON으로 통계 저장
        import json
        with open(os.path.join(cluster_dir, 'cluster_stats.json'), 'w') as f:
            json.dump(cluster_stats, f, indent=2, default=str)

def create_cluster_preview_grid(cluster_dir, image_names, cluster_id, result_dir):
    """클러스터별 이미지 미리보기 그리드 생성"""
    if not image_names:
        return
        
    # 5x10 그리드 (최대 50개)
    n_rows, n_cols = 5, 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(image_names):
            img_path = os.path.join(cluster_dir, image_names[i])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'{image_names[i][:15]}...', fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error\nloading', ha='center', va='center')
                print(f"Error loading {img_path}: {e}")
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f'Cluster {cluster_id} Sample Images (50 samples)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'cluster_{cluster_id}_preview.png'), 
                dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"     Created preview grid for Cluster {cluster_id}")

def analyze_tissue_distribution(results_df, result_dir):
    """조직별 클러스터 분포 분석"""
    print("🔬 Analyzing tissue distribution across clusters...")
    
    # 노이즈 제외
    valid_data = results_df[results_df['cluster'] != -1]
    
    if len(valid_data) == 0:
        print("   No valid clusters found")
        return
    
    # 크로스탭 생성
    crosstab = pd.crosstab(valid_data['tissue_index'], valid_data['cluster'], 
                          normalize='columns') * 100
    
    # 히트맵 생성
    plt.figure(figsize=(12, 6))
    sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='Blues', 
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Tissue Distribution Across Clusters (%)', fontsize=14)
    plt.xlabel('Cluster')
    plt.ylabel('Tissue Index')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'tissue_distribution_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 통계 테이블 저장
    crosstab.to_csv(os.path.join(result_dir, 'tissue_cluster_distribution.csv'))
    
    print("   Tissue distribution analysis saved")

# ============================================================================
#  실행 함수
# ============================================================================

def run_visualization():
    """메인 시각화 실행 함수"""
    try:
        create_cluster_visualization_and_samples()
    except Exception as e:
        print(f" Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 필요한 모든 함수들이 이미 정의되어 있다고 가정
    # (원본 코드에서 import)
    run_visualization()






import os
import matplotlib.pyplot as plt
from PIL import Image
import glob

def view_cluster(cluster_id, base_folder='clustering_results_2.0_500/cluster_samples', 
                 max_images=50, grid_size=(5, 10)):
    """
    특정 클러스터의 이미지들을 그리드로 보기
    
    Args:
        cluster_id: 클러스터 번호 (0, 1, 2, ...)
        base_folder: 클러스터 폴더들이 있는 기본 경로
        max_images: 최대 표시할 이미지 수
        grid_size: (행, 열) 튜플
    """
    cluster_folder = os.path.join(base_folder, f'cluster_{cluster_id}')
    
    if not os.path.exists(cluster_folder):
        print(f" 폴더를 찾을 수 없습니다: {cluster_folder}")
        return
    
    # 이미지 파일 찾기
    image_files = glob.glob(os.path.join(cluster_folder, '*.png'))
    image_files.sort()
    
    if not image_files:
        print(f" {cluster_folder}에서 이미지를 찾을 수 없습니다!")
        return
    
    print(f" 클러스터 {cluster_id}: {len(image_files)}개 이미지 발견")
    
    # 최대 개수로 제한
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    n_rows, n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    
    loaded_count = 0
    for i, ax in enumerate(axes):
        if i < len(image_files):
            try:
                img = Image.open(image_files[i])
                ax.imshow(img)
                filename = os.path.basename(image_files[i])
                ax.set_title(f'{filename[:10]}...', fontsize=6)
                loaded_count += 1
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                print(f" 로딩 실패: {image_files[i]}")
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f'🔬 Cluster {cluster_id} ({loaded_count} images)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def view_all_clusters(base_folder='clustering_results_2.0_500/cluster_samples', 
                     images_per_cluster=25):
    """
    모든 클러스터를 한번에 보기
    
    Args:
        base_folder: 클러스터 폴더들이 있는 기본 경로
        images_per_cluster: 각 클러스터당 표시할 이미지 수
    """
    if not os.path.exists(base_folder):
        print(f" 폴더를 찾을 수 없습니다: {base_folder}")
        return
    
    # 클러스터 폴더들 찾기
    cluster_folders = [f for f in os.listdir(base_folder) 
                      if f.startswith('cluster_') and os.path.isdir(os.path.join(base_folder, f))]
    cluster_folders.sort(key=lambda x: int(x.split('_')[1]))  # 번호순 정렬
    
    if not cluster_folders:
        print(f" {base_folder}에서 클러스터 폴더를 찾을 수 없습니다!")
        return
    
    print(f" {len(cluster_folders)}개 클러스터 발견: {cluster_folders}")
    
    # 각 클러스터별로 표시
    for folder in cluster_folders:
        cluster_id = folder.split('_')[1]
        print(f"\n{'='*50}")
        print(f" CLUSTER {cluster_id}")
        print('='*50)
        
        # 5x5 그리드로 25개씩 표시
        view_cluster(cluster_id, base_folder, max_images=images_per_cluster, grid_size=(5, 5))

def quick_overview(base_folder='clustering_results_5.0_500/cluster_samples'):
    """
    각 클러스터에서 대표 이미지 1개씩만 빠르게 보기
    """
    if not os.path.exists(base_folder):
        print(f" 폴더를 찾을 수 없습니다: {base_folder}")
        return
    
    # 클러스터 폴더들 찾기
    cluster_folders = [f for f in os.listdir(base_folder) 
                      if f.startswith('cluster_') and os.path.isdir(os.path.join(base_folder, f))]
    cluster_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    if not cluster_folders:
        print(f" 클러스터 폴더를 찾을 수 없습니다!")
        return
    
    n_clusters = len(cluster_folders)
    n_cols = min(n_clusters, 5)  # 최대 5열
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, folder in enumerate(cluster_folders):
        cluster_id = folder.split('_')[1]
        cluster_path = os.path.join(base_folder, folder)
        
        # 첫 번째 이미지 찾기
        image_files = glob.glob(os.path.join(cluster_path, '*.png'))
        
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        if image_files:
            try:
                img = Image.open(image_files[0])
                ax.imshow(img)
                ax.set_title(f'Cluster {cluster_id}\n({len(image_files)} images)', 
                           fontsize=10, fontweight='bold')
            except Exception:
                ax.text(0.5, 0.5, f'Cluster {cluster_id}\nError', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f'Cluster {cluster_id}\nNo images', ha='center', va='center')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 남은 subplot 숨기기
    for j in range(len(cluster_folders), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(' 클러스터 개요 (각 클러스터 대표 이미지)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 사용법 예시:
if __name__ == "__main__":
    print(" 클러스터 이미지 뷰어")
    print("="*40)
    print("사용법:")
    print("1. view_cluster(0)          # 클러스터 0만 보기")
    print("2. view_all_clusters()      # 모든 클러스터 보기")  
    print("3. quick_overview()         # 빠른 개요 보기")
    print("="*40)
    
    # 빠른 개요부터 시작
    quick_overview()



