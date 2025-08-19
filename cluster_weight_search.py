import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import cv2
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# GPU 메모리 관리
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# Positional Encoding 
# ============================================================================

def get_2d_positional_encoding(x, y, dim=64, max_len=10000):
    """
    2D sin/cos Positional Encoding
    
    Args:
        x, y: normalized coordinates (0~1 range)
        dim: encoding dimension (must be divisible by 4)
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D encoding"
    
    pos_encoding = np.zeros((len(x), dim))
    d_model_quarter = dim // 4
    div_term = np.exp(np.arange(0, d_model_quarter) * -(np.log(max_len) / d_model_quarter))
    
    # X coordinate encoding
    pos_encoding[:, 0::4] = np.sin(x[:, None] * div_term)
    pos_encoding[:, 1::4] = np.cos(x[:, None] * div_term)
    
    # Y coordinate encoding  
    pos_encoding[:, 2::4] = np.sin(y[:, None] * div_term)
    pos_encoding[:, 3::4] = np.cos(y[:, None] * div_term)
    
    return pos_encoding

def create_positional_features(df, pos_dim=64):
    """
    Create positional encoding from array coordinates (tissue별 상대좌표)
    """
    print("Creating positional encoding features...")
    
    pos_features_list = []
    
    # Tissue별로 정규화
    for tissue_id in df['tissue_index'].unique():
        tissue_mask = df['tissue_index'] == tissue_id
        tissue_data = df[tissue_mask]
        
        # Array 좌표 정규화 (0~1)
        x_coords = tissue_data['array_row'].values
        y_coords = tissue_data['array_col'].values
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        if x_max > x_min and y_max > y_min:
            x_norm = (x_coords - x_min) / (x_max - x_min)
            y_norm = (y_coords - y_min) / (y_max - y_min)
        else:
            x_norm = np.zeros_like(x_coords)
            y_norm = np.zeros_like(y_coords)
        
        # Positional encoding 생성
        tissue_pos = get_2d_positional_encoding(x_norm, y_norm, dim=pos_dim)
        pos_features_list.append(tissue_pos)
    
    pos_features = np.concatenate(pos_features_list)
    
    print(f" Positional encoding created: {pos_features.shape}")
    return pos_features

# ============================================================================
# Dataset & Feature Extractors
# ============================================================================

class TissueDataset(Dataset):
    def __init__(self, image_dir, df, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.df = df
        
        # ID에서 이미지 파일명
        self.image_files = [f"{row['id']}.png" for _, row in df.iterrows()]
        print(f"Dataset loaded: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name))
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# ============================================================================
# 조직병리 특화 ResNet18 로더
# ============================================================================

def load_histology_resnet18(model_path='weights/tenpercent_resnet18.ckpt', device='cuda'):
    """
    조직병리학 특화 ResNet18 모델 로드
    """
    print(f" Loading histology-specialized ResNet18 from {model_path}")
    
    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print(' No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model
    
    # ResNet18 모델 생성 (pretrained=False)
    model = models.resnet18(pretrained=False)
    
    try:
        # 체크포인트 로드
        state = torch.load(model_path, map_location=device)
        state_dict = state['state_dict']
        
        # 키 이름 정리 (model., resnet. 제거)
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        # 가중치 로드
        model = load_model_weights(model, state_dict)
        
        # FC layer를 Identity로 변경 (특징 추출용)
        model.fc = nn.Identity()
        
        model = model.to(device)
        model.eval()
        
        print(" Histology-specialized ResNet18 loaded successfully!")
        return model
        
    except Exception as e:
        print(f" Error loading histology weights: {e}")
        print(" Falling back to ImageNet pretrained ResNet18...")
        
        # ImageNet 사전훈련 모델로 폴백
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model = model.to(device)
        model.eval()
        
        return model

class HistologyFeatureExtractor:
    def __init__(self, model_path='weights/tenpercent_resnet18.ckpt', 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        
        # 조직병리학 특화 ResNet18 로드
        self.model = load_histology_resnet18(model_path, device)

    def extract_features(self, dataloader):
        """조직병리학 ResNet18 특징 추출"""
        resnet_features = []
        names = []
        
        with torch.no_grad():
            for images, img_names in tqdm(dataloader, desc="Extracting histology features"):
                images = images.to(self.device)
                
                # 조직병리학 특화 ResNet 특징
                batch_resnet = self.model(images)
                
                resnet_features.append(batch_resnet.cpu().numpy())
                names.extend(img_names)
                
                if len(resnet_features) % 10 == 0:
                    clear_gpu_memory()
        
        resnet_features = np.concatenate(resnet_features)
        
        print(f" Histology feature extraction complete:")
        print(f"   - ResNet features: {resnet_features.shape}")
        
        return resnet_features, names

# ============================================================================
#  Feature Fusion & Clustering
# ============================================================================

def fuse_features(resnet_features, pos_features, 
                 resnet_weight=1.0, pos_weight=100.0):
    """
    특징 융합: Histology ResNet + Positional Encoding
    """
    # L2 정규화 후 가중치 적용
    resnet_norm = resnet_features / np.linalg.norm(resnet_features, axis=1, keepdims=True)
    pos_norm = pos_features / np.linalg.norm(pos_features, axis=1, keepdims=True)
    
    # 가중치 적용 후 결합
    fused_features = np.concatenate([
        resnet_norm * resnet_weight,
        pos_norm * pos_weight
    ], axis=1)
    
    return fused_features

def evaluate_clustering(features, clusters):
    """클러스터링 성능 평가"""
    if len(set(clusters)) <= 1:  # 클러스터가 1개 이하면 평가 불가
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': 0}
    
    # 노이즈 포인트 제거
    valid_mask = clusters != -1
    if np.sum(valid_mask) < 2:
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': 0}
    
    valid_features = features[valid_mask]
    valid_clusters = clusters[valid_mask]
    
    if len(set(valid_clusters)) <= 1:
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': len(set(valid_clusters))}
    
    try:
        silhouette = silhouette_score(valid_features, valid_clusters)
        calinski = calinski_harabasz_score(valid_features, valid_clusters)
        davies = davies_bouldin_score(valid_features, valid_clusters)
        n_clusters = len(set(valid_clusters))
        
        return {
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'n_clusters': n_clusters
        }
    except:
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': 0}

def perform_clustering_with_eval(features):
    """UMAP + HDBSCAN 클러스터링 및 평가"""
    # 특징 정규화
    features_scaled = StandardScaler().fit_transform(features)
    
    # UMAP 차원 축소
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.001,
        n_components=2,
        random_state=42,
        metric='euclidean'
    )
    umap_embedded = reducer.fit_transform(features_scaled)
    
    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=70,
        min_samples=3,
        cluster_selection_epsilon=0.05,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(umap_embedded)
    
    # 성능 평가
    metrics = evaluate_clustering(umap_embedded, clusters)
    
    return umap_embedded, clusters, metrics

# ============================================================================
#  가중치 조합 최적화 및 시각화
# ============================================================================

def comprehensive_weight_optimization(resnet_features, pos_features, save_dir='.'):
    """
    가중치 조합에 대해 클러스터링 수행 및 UMAP 시각화
    """
    print(" Starting weight optimization...")
    
    # 2차원 가중치 조합 정의
    resnet_weights = [0.1, 0.5, 1.0, 2.0, 5.0]
    pos_weights = [1.0, 10.0, 50.0, 100.0, 500.0]  
    
    print(f"Testing {len(resnet_weights)} × {len(pos_weights)} = {len(resnet_weights) * len(pos_weights)} combinations")
    
    # 결과 저장
    all_results = []
    
    combinations = list(product(resnet_weights, pos_weights))
    
    for i, (rw, pw) in enumerate(tqdm(combinations, desc="Testing weight combinations")):
        try:
            # 특징 융합
            fused_features = fuse_features(resnet_features, pos_features,
                                         resnet_weight=rw, pos_weight=pw)
            
            # 클러스터링 및 평가
            umap_embedded, clusters, metrics = perform_clustering_with_eval(fused_features)
            
            result = {
                'combination_idx': i,
                'resnet_weight': rw,
                'pos_weight': pw,  
                'silhouette': metrics['silhouette'],
                'calinski': metrics['calinski'],
                'davies': metrics['davies'],
                'n_clusters': metrics['n_clusters'],
                'umap_embedded': umap_embedded,
                'clusters': clusters
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error with combination ({rw}, {pw}): {e}")
            continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['umap_embedded', 'clusters']}
        for r in all_results
    ])
    
    # 결과 저장
    results_df.to_csv(os.path.join(save_dir, 'weight_optimization_results.csv'), index=False)
    
    # 최적 조합들 선택
    top_results = sorted(all_results, key=lambda x: x['silhouette'], reverse=True)[:20]
    
    print(f"\n Top 20 combinations found!")
    print("="*60)
    for i, result in enumerate(top_results[:10]):
        print(f"{i+1:2d}. ResNet:{result['resnet_weight']:4.1f} | Pos:{result['pos_weight']:5.0f} | "
              f"Sil:{result['silhouette']:.3f} | Clusters:{result['n_clusters']:2d}")
    
    #  종합 시각화
    plot_comprehensive_results(top_results, results_df, save_dir)
    
    return top_results, results_df

def plot_comprehensive_results(top_results, results_df, save_dir):
    """종합 결과 시각화"""
    
    # 1.  상위 20개 조합의 UMAP 결과 (4x5 그리드)
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()
    
    for i, result in enumerate(top_results):
        ax = axes[i]
        
        umap_embedded = result['umap_embedded']
        clusters = result['clusters']
        
        # 노이즈 포인트
        noise_mask = clusters == -1
        if np.any(noise_mask):
            ax.scatter(umap_embedded[noise_mask, 0], umap_embedded[noise_mask, 1],
                      c='lightgray', alpha=0.3, s=8, label='Noise')
        
        # 클러스터 포인트
        cluster_mask = ~noise_mask
        if np.any(cluster_mask):
            scatter = ax.scatter(umap_embedded[cluster_mask, 0], umap_embedded[cluster_mask, 1],
                               c=clusters[cluster_mask], cmap='tab20', alpha=0.8, s=12)
        
        ax.set_title(f'#{i+1}: ({result["resnet_weight"]:.1f}, {result["pos_weight"]:.0f})\n'
                    f'Sil: {result["silhouette"]:.3f} | Clusters: {result["n_clusters"]}', 
                    fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(' Top 20 Weight Combinations - UMAP Clustering Results', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'top20_umap_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 성능 메트릭 분석
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 실루엣 점수 분포
    axes[0, 0].hist(results_df['silhouette'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(results_df['silhouette'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["silhouette"].mean():.3f}')
    axes[0, 0].set_title('Silhouette Score Distribution')
    axes[0, 0].set_xlabel('Silhouette Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 클러스터 수 분포
    max_clusters = results_df['n_clusters'].max()
    axes[0, 1].hist(results_df['n_clusters'], bins=range(0, max_clusters+2), 
                    alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Number of Clusters Distribution')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ResNet 가중치별 성능
    resnet_perf = results_df.groupby('resnet_weight')['silhouette'].agg(['mean', 'std']).reset_index()
    axes[0, 2].bar(resnet_perf['resnet_weight'], resnet_perf['mean'], 
                   yerr=resnet_perf['std'], capsize=5, alpha=0.7, color='orange')
    axes[0, 2].set_title('Performance by ResNet Weight')
    axes[0, 2].set_xlabel('ResNet Weight')
    axes[0, 2].set_ylabel('Mean Silhouette Score')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Positional 가중치별 성능
    pos_perf = results_df.groupby('pos_weight')['silhouette'].agg(['mean', 'std']).reset_index()
    axes[1, 0].bar(range(len(pos_perf)), pos_perf['mean'], 
                   yerr=pos_perf['std'], capsize=5, alpha=0.7, color='red')
    axes[1, 0].set_title('Performance by Positional Weight')
    axes[1, 0].set_xlabel('Positional Weight')
    axes[1, 0].set_ylabel('Mean Silhouette Score')
    axes[1, 0].set_xticks(range(len(pos_perf)))
    axes[1, 0].set_xticklabels([f'{w:.0f}' for w in pos_perf['pos_weight']], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 가중치 조합 히트맵 (ResNet vs Pos)
    pivot_data = results_df.groupby(['resnet_weight', 'pos_weight'])['silhouette'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Weight Combination Heatmap\n(ResNet vs Positional)')
    axes[1, 1].set_xlabel('Positional Weight')
    axes[1, 1].set_ylabel('ResNet Weight')
    
    # 빈 subplot 제거
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_optimization_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
#  MAIN 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("🔬 Histology-Specialized Clustering with Weight Optimization")
    print("="*60)
    
    # 데이터 로드
    print(" Loading data...")
    df = pd.read_csv('dataset.csv')
    
    #  조직 3번 제외 (1, 2, 4번만 사용)
    print(f"Original data: {len(df)} samples")
    df = df[df['tissue_index'].isin([1, 2, 4])]
    print(f"After filtering (tissues 1,2,4 only): {len(df)} samples")
    print(f"Tissue distribution: {df['tissue_index'].value_counts().sort_index()}")
    
    # Transform 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset & DataLoader
    dataset = TissueDataset('train', df, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    #  1. Positional Encoding 생성
    print("\nStep 1: Creating positional encoding...")
    pos_features = create_positional_features(df, pos_dim=64)
    
    # 2. 조직병리학 특화 ResNet 특징 추출
    print("\n Step 2: Extracting histology-specialized features...")
    extractor = HistologyFeatureExtractor(model_path='weights/tenpercent_resnet18.ckpt')
    resnet_features, names = extractor.extract_features(dataloader)
    
    # 3. 가중치 최적화 및 시각화
    print("\n Step 3: Weight optimization...")
    top_results, results_df = comprehensive_weight_optimization(
        resnet_features, pos_features, save_dir='.'
    )
    
    #  4. 최적 결과로 최종 클러스터링
    print("\n Step 4: Final clustering with best weights...")
    best_result = top_results[0]
    
    #  5. 결과 저장
    print("\n Step 5: Saving final results...")
    final_results_df = pd.DataFrame({
        'image_name': names,
        'cluster': best_result['clusters'],
        'array_row': df['array_row'].values,
        'array_col': df['array_col'].values,
        'tissue_index': df['tissue_index'].values,
        'umap_x': best_result['umap_embedded'][:, 0],
        'umap_y': best_result['umap_embedded'][:, 1]
    })
    
    final_results_df.to_csv('histology_clustering_results.csv', index=False)
    
    # 최종 통계
    cluster_stats = final_results_df['cluster'].value_counts().sort_index()
    print(f"\n Final Clustering Statistics (Best Combination):")
    print(f"   Best weights: ResNet={best_result['resnet_weight']:.1f}, "
          f"Pos={best_result['pos_weight']:.0f}")
    print(f"   Silhouette Score: {best_result['silhouette']:.3f}")
    print(f"   Number of clusters: {best_result['n_clusters']}")
    print("   Cluster distribution:")
    for cluster_id, count in cluster_stats.items():
        if cluster_id == -1:
            print(f"     Noise: {count} samples")
        else:
            print(f"     Cluster {cluster_id}: {count} samples")
    
    print("\n🎉 Complete! Results saved:")
    print("   - histology_clustering_results.csv")
    print("   - weight_optimization_results.csv") 
    print("   - top20_umap_results.png")
    print("   - weight_optimization_analysis.png")

# ============================================================================
#  빠른 테스트 함수
# ============================================================================

def run_quick_test():
    """빠른 테스트 (소수 조합만)"""
    print(" Running Quick Test...")
    
    # 데이터 로드
    df = pd.read_csv('dataset.csv')
    
    #  조직 3번 제외 (1, 2, 4번만 사용)  
    df = df[df['tissue_index'].isin([1, 2, 4])]
    print(f"Using tissues 1, 2, 4 only: {len(df)} samples")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TissueDataset('train', df, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 특징 추출
    pos_features = create_positional_features(df, pos_dim=64)
    extractor = HistologyFeatureExtractor(model_path='weights/tenpercent_resnet18.ckpt')
    resnet_features, names = extractor.extract_features(dataloader)
    
    # 제한된 조합 테스트
    quick_combinations = [
        (0.5, 50), (1.0, 100), (2.0, 100), (1.0, 500)
    ]
    
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (rw, pw) in enumerate(quick_combinations):
        print(f"Testing combination {i+1}/4: ({rw}, {pw})")
        
        fused_features = fuse_features(resnet_features, pos_features,
                                     resnet_weight=rw, pos_weight=pw)
        
        umap_embedded, clusters, metrics = perform_clustering_with_eval(fused_features)
        
        # 시각화
        ax = axes[i]
        noise_mask = clusters == -1
        if np.any(noise_mask):
            ax.scatter(umap_embedded[noise_mask, 0], umap_embedded[noise_mask, 1],
                      c='lightgray', alpha=0.3, s=10)
        
        cluster_mask = ~noise_mask
        if np.any(cluster_mask):
            ax.scatter(umap_embedded[cluster_mask, 0], umap_embedded[cluster_mask, 1],
                      c=clusters[cluster_mask], cmap='tab20', alpha=0.8, s=15)
        
        ax.set_title(f'Weights: ({rw}, {pw})\nSil: {metrics["silhouette"]:.3f} | '
                    f'Clusters: {metrics["n_clusters"]}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        results.append((rw, pw, metrics['silhouette'], metrics['n_clusters']))
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 출력
    print("\n Quick Test Results:")
    results.sort(key=lambda x: x[2], reverse=True)
    for i, (rw, pw, sil, n_clust) in enumerate(results):
        print(f"{i+1}. ({rw:3.1f}, {pw:3.0f}) | Sil: {sil:.3f} | Clusters: {n_clust}")

if __name__ == "__main__":
    # 실행 옵션 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_test()
    else:
        # 기본 실행
        main()

=================
def show_gallery(d, max_imgs=200, size=224):
    imgs = list_images(d)[:max_imgs]
    html = ["<div style='display:flex;flex-wrap:wrap;gap:6px'>"]
    for p in imgs:
        html.append(f"<img src='{p.as_posix()}' width='{size}' height='{size}'/>")
    html.append("</div>")
    display(HTML("".join(html)))

for d in clusters:
    print(d.name)
    show_gallery(d, max_imgs=300, size=224)  

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import cv2
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# GPU 메모리 관리
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# Positional Encoding 
# ============================================================================

def get_2d_positional_encoding(x, y, dim=64, max_len=10000):
    """
    2D sin/cos Positional Encoding
    
    Args:
        x, y: normalized coordinates (0~1 range)
        dim: encoding dimension (must be divisible by 4)
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D encoding"
    
    pos_encoding = np.zeros((len(x), dim))
    d_model_quarter = dim // 4
    div_term = np.exp(np.arange(0, d_model_quarter) * -(np.log(max_len) / d_model_quarter))
    
    # X coordinate encoding
    pos_encoding[:, 0::4] = np.sin(x[:, None] * div_term)
    pos_encoding[:, 1::4] = np.cos(x[:, None] * div_term)
    
    # Y coordinate encoding  
    pos_encoding[:, 2::4] = np.sin(y[:, None] * div_term)
    pos_encoding[:, 3::4] = np.cos(y[:, None] * div_term)
    
    return pos_encoding

def create_positional_features(df, pos_dim=64):
    """
    Create positional encoding from array coordinates (tissue별 상대좌표)
    """
    print("Creating positional encoding features...")
    
    pos_features_list = []
    
    # Tissue별로 정규화
    for tissue_id in df['tissue_index'].unique():
        tissue_mask = df['tissue_index'] == tissue_id
        tissue_data = df[tissue_mask]
        
        # Array 좌표 정규화 (0~1)
        x_coords = tissue_data['array_row'].values
        y_coords = tissue_data['array_col'].values
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        if x_max > x_min and y_max > y_min:
            x_norm = (x_coords - x_min) / (x_max - x_min)
            y_norm = (y_coords - y_min) / (y_max - y_min)
        else:
            x_norm = np.zeros_like(x_coords)
            y_norm = np.zeros_like(y_coords)
        
        # Positional encoding 생성
        tissue_pos = get_2d_positional_encoding(x_norm, y_norm, dim=pos_dim)
        pos_features_list.append(tissue_pos)
    
    pos_features = np.concatenate(pos_features_list)
    
    print(f" Positional encoding created: {pos_features.shape}")
    return pos_features

# ============================================================================
# Dataset & Feature Extractors
# ============================================================================

class TissueDataset(Dataset):
    def __init__(self, image_dir, df, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.df = df
        
        # ID에서 이미지 파일명
        self.image_files = [f"{row['id']}.png" for _, row in df.iterrows()]
        print(f"Dataset loaded: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name))
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# ============================================================================
# 조직병리 특화 ResNet18 로더
# ============================================================================

def load_histology_resnet18(model_path='weights/tenpercent_resnet18.ckpt', device='cuda'):
    """
    조직병리학 특화 ResNet18 모델 로드
    """
    print(f" Loading histology-specialized ResNet18 from {model_path}")
    
    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print(' No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model
    
    # ResNet18 모델 생성 (pretrained=False)
    model = models.resnet18(pretrained=False)
    
    try:
        # 체크포인트 로드
        state = torch.load(model_path, map_location=device)
        state_dict = state['state_dict']
        
        # 키 이름 정리 (model., resnet. 제거)
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        # 가중치 로드
        model = load_model_weights(model, state_dict)
        
        # FC layer를 Identity로 변경 (특징 추출용)
        model.fc = nn.Identity()
        
        model = model.to(device)
        model.eval()
        
        print(" Histology-specialized ResNet18 loaded successfully!")
        return model
        
    except Exception as e:
        print(f" Error loading histology weights: {e}")
        print(" Falling back to ImageNet pretrained ResNet18...")
        
        # ImageNet 사전훈련 모델로 폴백
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model = model.to(device)
        model.eval()
        
        return model

class HistologyFeatureExtractor:
    def __init__(self, model_path='weights/tenpercent_resnet18.ckpt', 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        
        # 조직병리학 특화 ResNet18 로드
        self.model = load_histology_resnet18(model_path, device)

    def extract_features(self, dataloader):
        """조직병리학 ResNet18 특징 추출"""
        resnet_features = []
        names = []
        
        with torch.no_grad():
            for images, img_names in tqdm(dataloader, desc="Extracting histology features"):
                images = images.to(self.device)
                
                # 조직병리학 특화 ResNet 특징
                batch_resnet = self.model(images)
                
                resnet_features.append(batch_resnet.cpu().numpy())
                names.extend(img_names)
                
                if len(resnet_features) % 10 == 0:
                    clear_gpu_memory()
        
        resnet_features = np.concatenate(resnet_features)
        
        print(f" Histology feature extraction complete:")
        print(f"   - ResNet features: {resnet_features.shape}")
        
        return resnet_features, names

# ============================================================================
#  Feature Fusion & Clustering
# ============================================================================

def fuse_features(resnet_features, pos_features, 
                 resnet_weight=1.0, pos_weight=100.0):
    """
    특징 융합: Histology ResNet + Positional Encoding
    """
    # L2 정규화 후 가중치 적용
    resnet_norm = resnet_features / np.linalg.norm(resnet_features, axis=1, keepdims=True)
    pos_norm = pos_features / np.linalg.norm(pos_features, axis=1, keepdims=True)
    
    # 가중치 적용 후 결합
    fused_features = np.concatenate([
        resnet_norm * resnet_weight,
        pos_norm * pos_weight
    ], axis=1)
    
    return fused_features

def evaluate_clustering(features, clusters):
    """클러스터링 성능 평가"""
    if len(set(clusters)) <= 1:  # 클러스터가 1개 이하면 평가 불가
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': 0}
    
    # 노이즈 포인트 제거
    valid_mask = clusters != -1
    if np.sum(valid_mask) < 2:
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': 0}
    
    valid_features = features[valid_mask]
    valid_clusters = clusters[valid_mask]
    
    if len(set(valid_clusters)) <= 1:
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': len(set(valid_clusters))}
    
    try:
        silhouette = silhouette_score(valid_features, valid_clusters)
        calinski = calinski_harabasz_score(valid_features, valid_clusters)
        davies = davies_bouldin_score(valid_features, valid_clusters)
        n_clusters = len(set(valid_clusters))
        
        return {
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'n_clusters': n_clusters
        }
    except:
        return {'silhouette': -1, 'calinski': 0, 'davies': float('inf'), 'n_clusters': 0}

def perform_clustering_with_eval(features):
    """UMAP + HDBSCAN 클러스터링 및 평가"""
    # 특징 정규화
    features_scaled = StandardScaler().fit_transform(features)
    
    # UMAP 차원 축소
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.001,
        n_components=2,
        random_state=42,
        metric='euclidean'
    )
    umap_embedded = reducer.fit_transform(features_scaled)
    
    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=70,
        min_samples=3,
        cluster_selection_epsilon=0.05,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(umap_embedded)
    
    # 성능 평가
    metrics = evaluate_clustering(umap_embedded, clusters)
    
    return umap_embedded, clusters, metrics

# ============================================================================
#  가중치 조합 최적화 및 시각화
# ============================================================================

def comprehensive_weight_optimization(resnet_features, pos_features, save_dir='.'):
    """
    가중치 조합에 대해 클러스터링 수행 및 UMAP 시각화
    """
    print(" Starting weight optimization...")
    
    # 2차원 가중치 조합 정의
    resnet_weights = [0.1, 0.5, 1.0, 2.0, 5.0]
    pos_weights = [1.0, 10.0, 50.0, 100.0, 500.0]  
    
    print(f"Testing {len(resnet_weights)} × {len(pos_weights)} = {len(resnet_weights) * len(pos_weights)} combinations")
    
    # 결과 저장
    all_results = []
    
    combinations = list(product(resnet_weights, pos_weights))
    
    for i, (rw, pw) in enumerate(tqdm(combinations, desc="Testing weight combinations")):
        try:
            # 특징 융합
            fused_features = fuse_features(resnet_features, pos_features,
                                         resnet_weight=rw, pos_weight=pw)
            
            # 클러스터링 및 평가
            umap_embedded, clusters, metrics = perform_clustering_with_eval(fused_features)
            
            result = {
                'combination_idx': i,
                'resnet_weight': rw,
                'pos_weight': pw,  
                'silhouette': metrics['silhouette'],
                'calinski': metrics['calinski'],
                'davies': metrics['davies'],
                'n_clusters': metrics['n_clusters'],
                'umap_embedded': umap_embedded,
                'clusters': clusters
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error with combination ({rw}, {pw}): {e}")
            continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['umap_embedded', 'clusters']}
        for r in all_results
    ])
    
    # 결과 저장
    results_df.to_csv(os.path.join(save_dir, 'weight_optimization_results.csv'), index=False)
    
    # 최적 조합들 선택
    top_results = sorted(all_results, key=lambda x: x['silhouette'], reverse=True)[:20]
    
    print(f"\n Top 20 combinations found!")
    print("="*60)
    for i, result in enumerate(top_results[:10]):
        print(f"{i+1:2d}. ResNet:{result['resnet_weight']:4.1f} | Pos:{result['pos_weight']:5.0f} | "
              f"Sil:{result['silhouette']:.3f} | Clusters:{result['n_clusters']:2d}")
    
    #  종합 시각화
    plot_comprehensive_results(top_results, results_df, save_dir)
    
    return top_results, results_df

def plot_comprehensive_results(top_results, results_df, save_dir):
    """종합 결과 시각화"""
    
    # 1.  상위 20개 조합의 UMAP 결과 (4x5 그리드)
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()
    
    for i, result in enumerate(top_results):
        ax = axes[i]
        
        umap_embedded = result['umap_embedded']
        clusters = result['clusters']
        
        # 노이즈 포인트
        noise_mask = clusters == -1
        if np.any(noise_mask):
            ax.scatter(umap_embedded[noise_mask, 0], umap_embedded[noise_mask, 1],
                      c='lightgray', alpha=0.3, s=8, label='Noise')
        
        # 클러스터 포인트
        cluster_mask = ~noise_mask
        if np.any(cluster_mask):
            scatter = ax.scatter(umap_embedded[cluster_mask, 0], umap_embedded[cluster_mask, 1],
                               c=clusters[cluster_mask], cmap='tab20', alpha=0.8, s=12)
        
        ax.set_title(f'#{i+1}: ({result["resnet_weight"]:.1f}, {result["pos_weight"]:.0f})\n'
                    f'Sil: {result["silhouette"]:.3f} | Clusters: {result["n_clusters"]}', 
                    fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(' Top 20 Weight Combinations - UMAP Clustering Results', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'top20_umap_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 성능 메트릭 분석
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 실루엣 점수 분포
    axes[0, 0].hist(results_df['silhouette'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(results_df['silhouette'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["silhouette"].mean():.3f}')
    axes[0, 0].set_title('Silhouette Score Distribution')
    axes[0, 0].set_xlabel('Silhouette Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 클러스터 수 분포
    max_clusters = results_df['n_clusters'].max()
    axes[0, 1].hist(results_df['n_clusters'], bins=range(0, max_clusters+2), 
                    alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Number of Clusters Distribution')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ResNet 가중치별 성능
    resnet_perf = results_df.groupby('resnet_weight')['silhouette'].agg(['mean', 'std']).reset_index()
    axes[0, 2].bar(resnet_perf['resnet_weight'], resnet_perf['mean'], 
                   yerr=resnet_perf['std'], capsize=5, alpha=0.7, color='orange')
    axes[0, 2].set_title('Performance by ResNet Weight')
    axes[0, 2].set_xlabel('ResNet Weight')
    axes[0, 2].set_ylabel('Mean Silhouette Score')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Positional 가중치별 성능
    pos_perf = results_df.groupby('pos_weight')['silhouette'].agg(['mean', 'std']).reset_index()
    axes[1, 0].bar(range(len(pos_perf)), pos_perf['mean'], 
                   yerr=pos_perf['std'], capsize=5, alpha=0.7, color='red')
    axes[1, 0].set_title('Performance by Positional Weight')
    axes[1, 0].set_xlabel('Positional Weight')
    axes[1, 0].set_ylabel('Mean Silhouette Score')
    axes[1, 0].set_xticks(range(len(pos_perf)))
    axes[1, 0].set_xticklabels([f'{w:.0f}' for w in pos_perf['pos_weight']], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 가중치 조합 히트맵 (ResNet vs Pos)
    pivot_data = results_df.groupby(['resnet_weight', 'pos_weight'])['silhouette'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Weight Combination Heatmap\n(ResNet vs Positional)')
    axes[1, 1].set_xlabel('Positional Weight')
    axes[1, 1].set_ylabel('ResNet Weight')
    
    # 빈 subplot 제거
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_optimization_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
#  MAIN 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("🔬 Histology-Specialized Clustering with Weight Optimization")
    print("="*60)
    
    # 데이터 로드
    print(" Loading data...")
    df = pd.read_csv('dataset.csv')
    
    #  조직 3번 제외 (1, 2, 4번만 사용)
    print(f"Original data: {len(df)} samples")
    df = df[df['tissue_index'].isin([1, 2, 4])]
    print(f"After filtering (tissues 1,2,4 only): {len(df)} samples")
    print(f"Tissue distribution: {df['tissue_index'].value_counts().sort_index()}")
    
    # Transform 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset & DataLoader
    dataset = TissueDataset('train', df, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    #  1. Positional Encoding 생성
    print("\nStep 1: Creating positional encoding...")
    pos_features = create_positional_features(df, pos_dim=64)
    
    # 2. 조직병리학 특화 ResNet 특징 추출
    print("\n Step 2: Extracting histology-specialized features...")
    extractor = HistologyFeatureExtractor(model_path='weights/tenpercent_resnet18.ckpt')
    resnet_features, names = extractor.extract_features(dataloader)
    
    # 3. 가중치 최적화 및 시각화
    print("\n Step 3: Weight optimization...")
    top_results, results_df = comprehensive_weight_optimization(
        resnet_features, pos_features, save_dir='.'
    )
    
    #  4. 최적 결과로 최종 클러스터링
    print("\n Step 4: Final clustering with best weights...")
    best_result = top_results[0]
    
    #  5. 결과 저장
    print("\n Step 5: Saving final results...")
    final_results_df = pd.DataFrame({
        'image_name': names,
        'cluster': best_result['clusters'],
        'array_row': df['array_row'].values,
        'array_col': df['array_col'].values,
        'tissue_index': df['tissue_index'].values,
        'umap_x': best_result['umap_embedded'][:, 0],
        'umap_y': best_result['umap_embedded'][:, 1]
    })
    
    final_results_df.to_csv('histology_clustering_results.csv', index=False)
    
    # 최종 통계
    cluster_stats = final_results_df['cluster'].value_counts().sort_index()
    print(f"\n Final Clustering Statistics (Best Combination):")
    print(f"   Best weights: ResNet={best_result['resnet_weight']:.1f}, "
          f"Pos={best_result['pos_weight']:.0f}")
    print(f"   Silhouette Score: {best_result['silhouette']:.3f}")
    print(f"   Number of clusters: {best_result['n_clusters']}")
    print("   Cluster distribution:")
    for cluster_id, count in cluster_stats.items():
        if cluster_id == -1:
            print(f"     Noise: {count} samples")
        else:
            print(f"     Cluster {cluster_id}: {count} samples")
    
    print("\n🎉 Complete! Results saved:")
    print("   - histology_clustering_results.csv")
    print("   - weight_optimization_results.csv") 
    print("   - top20_umap_results.png")
    print("   - weight_optimization_analysis.png")

# ============================================================================
#  빠른 테스트 함수
# ============================================================================

def run_quick_test():
    """빠른 테스트 (소수 조합만)"""
    print(" Running Quick Test...")
    
    # 데이터 로드
    df = pd.read_csv('dataset.csv')
    
    #  조직 3번 제외 (1, 2, 4번만 사용)  
    df = df[df['tissue_index'].isin([1, 2, 4])]
    print(f"Using tissues 1, 2, 4 only: {len(df)} samples")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TissueDataset('train', df, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 특징 추출
    pos_features = create_positional_features(df, pos_dim=64)
    extractor = HistologyFeatureExtractor(model_path='weights/tenpercent_resnet18.ckpt')
    resnet_features, names = extractor.extract_features(dataloader)
    
    # 제한된 조합 테스트
    quick_combinations = [
        (0.5, 50), (1.0, 100), (2.0, 100), (1.0, 500)
    ]
    
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (rw, pw) in enumerate(quick_combinations):
        print(f"Testing combination {i+1}/4: ({rw}, {pw})")
        
        fused_features = fuse_features(resnet_features, pos_features,
                                     resnet_weight=rw, pos_weight=pw)
        
        umap_embedded, clusters, metrics = perform_clustering_with_eval(fused_features)
        
        # 시각화
        ax = axes[i]
        noise_mask = clusters == -1
        if np.any(noise_mask):
            ax.scatter(umap_embedded[noise_mask, 0], umap_embedded[noise_mask, 1],
                      c='lightgray', alpha=0.3, s=10)
        
        cluster_mask = ~noise_mask
        if np.any(cluster_mask):
            ax.scatter(umap_embedded[cluster_mask, 0], umap_embedded[cluster_mask, 1],
                      c=clusters[cluster_mask], cmap='tab20', alpha=0.8, s=15)
        
        ax.set_title(f'Weights: ({rw}, {pw})\nSil: {metrics["silhouette"]:.3f} | '
                    f'Clusters: {metrics["n_clusters"]}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        results.append((rw, pw, metrics['silhouette'], metrics['n_clusters']))
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 출력
    print("\n Quick Test Results:")
    results.sort(key=lambda x: x[2], reverse=True)
    for i, (rw, pw, sil, n_clust) in enumerate(results):
        print(f"{i+1}. ({rw:3.1f}, {pw:3.0f}) | Sil: {sil:.3f} | Clusters: {n_clust}")

if __name__ == "__main__":
    # 실행 옵션 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_test()
    else:
        # 기본 실행
        main()

=================
def show_gallery(d, max_imgs=200, size=224):
    imgs = list_images(d)[:max_imgs]
    html = ["<div style='display:flex;flex-wrap:wrap;gap:6px'>"]
    for p in imgs:
        html.append(f"<img src='{p.as_posix()}' width='{size}' height='{size}'/>")
    html.append("</div>")
    display(HTML("".join(html)))

for d in clusters:
    print(d.name)
    show_gallery(d, max_imgs=300, size=224) 
