# -*- coding: utf-8 -*-
"""
特征提取模块 - 封装ResNet/PCA/t-SNE等特征提取方法
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, method='resnet', **kwargs):
        """
        初始化特征提取器
        Args:
            method: 特征提取方法 ('resnet', 'pca', 'tsne', 'raw')
            **kwargs: 方法特定参数
        """
        self.method = method
        self.kwargs = kwargs
    
    def extract_features(self, data):
        """
        提取特征
        Args:
            data: 输入数据 [N, 3072]
        Returns:
            features: 提取的特征 [N, feature_dim]
        """
        if isinstance(data, torch.Tensor):
            X_np = data.cpu().numpy()
        else:
            X_np = data.copy()
        
        if self.method == 'resnet':
            return self._extract_resnet_features(X_np)
        elif self.method == 'pca':
            return self._extract_pca_features(X_np)
        elif self.method == 'tsne':
            return self._extract_tsne_features(X_np)
        elif self.method == 'raw':
            return X_np
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")
    
    def _extract_resnet_features(self, X_np):
        """使用ResNet18提取特征"""
        print("使用ResNet18进行特征提取...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        import torchvision.models as models
        resnet_model = models.resnet18(pretrained=True)
        resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
        
        X_images = X_np.reshape(-1, 3, 32, 32)
        X_tensor = torch.FloatTensor(X_images).to(device)
        
        batch_size = self.kwargs.get('batch_size', 256)
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                features = resnet_model(batch)
                features = features.view(features.size(0), -1)
                features_list.append(features.cpu().numpy())
        
        X_features = np.vstack(features_list)
        print(f"ResNet特征提取完成，特征维度: {X_features.shape}")
        print(f"特征维度从 {X_np.shape[1]} 降至 {X_features.shape[1]}")
        return X_features
    
    def _extract_pca_features(self, X_np):
        """使用PCA提取特征"""
        print("使用PCA进行特征提取...")
        n_components = self.kwargs.get('n_components', 512)
        random_state = self.kwargs.get('random_state', 42)
        
        pca = PCA(n_components=n_components, random_state=random_state)
        X_features = pca.fit_transform(X_np)
        
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        print(f"PCA特征提取完成，特征维度: {X_features.shape}")
        print(f"特征维度从 {X_np.shape[1]} 降至 {X_features.shape[1]}")
        print(f"前{n_components}个主成分解释了 {explained_variance_ratio:.4f} ({explained_variance_ratio*100:.2f}%) 的方差")
        return X_features
    
    def _extract_tsne_features(self, X_np):
        """使用t-SNE提取特征"""
        print("使用t-SNE进行特征提取...")
        from sklearn.manifold import TSNE
        
        n_components = self.kwargs.get('n_components', 50)
        perplexity = self.kwargs.get('perplexity', 30)
        learning_rate = self.kwargs.get('learning_rate', 200)
        max_iter = self.kwargs.get('max_iter', 1000)
        random_state = self.kwargs.get('random_state', 42)
        
        if X_np.shape[1] > 100:
            print("数据维度较高，先使用PCA降至100维以加速t-SNE计算...")
            pca_pre = PCA(n_components=100, random_state=random_state)
            X_pca = pca_pre.fit_transform(X_np)
            pca_var_ratio = np.sum(pca_pre.explained_variance_ratio_)
            print(f"PCA预处理: 保留了 {pca_var_ratio:.4f} ({pca_var_ratio*100:.2f}%) 的方差")
        else:
            X_pca = X_np
        
        if len(X_pca) > 10000:
            print(f"数据量较大({len(X_pca)}样本)，为加速计算，使用前10000个样本进行t-SNE...")
            subset_indices = np.random.choice(len(X_pca), 10000, replace=False)
            X_subset = X_pca[subset_indices]
            
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state,
                init='random',
                method='barnes_hut',
                verbose=1
            )
            
            X_tsne_subset = tsne.fit_transform(X_subset)
            
            print("使用k-NN将剩余数据映射到t-SNE空间...")
            from sklearn.neighbors import NearestNeighbors
            
            k = min(5, len(X_subset))
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(X_subset)
            
            X_features = np.zeros((len(X_pca), n_components))
            X_features[subset_indices] = X_tsne_subset
            
            remaining_mask = np.ones(len(X_pca), dtype=bool)
            remaining_mask[subset_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) > 0:
                X_remaining = X_pca[remaining_indices]
                distances, indices = knn.kneighbors(X_remaining)
                
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                for i, idx in enumerate(remaining_indices):
                    neighbor_features = X_tsne_subset[indices[i]]
                    X_features[idx] = np.average(neighbor_features, weights=weights[i], axis=0)
        else:
            print("正在计算t-SNE特征...")
            tsne = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, (len(X_pca) - 1) // 3),
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state,
                init='random',
                method='barnes_hut' if len(X_pca) > 1000 else 'exact',
                verbose=1
            )
            
            X_features = tsne.fit_transform(X_pca)
        
        print(f"t-SNE特征提取完成，特征维度: {X_features.shape}")
        print(f"特征维度从 {X_np.shape[1]} 降至 {X_features.shape[1]}")
        return X_features


class SampleSelector:
    """样本选择器 - 基于K-means聚类选择样本"""
    
    def __init__(self, n_clusters=100, n_init=10, max_iter=300, random_state=42):
        """
        初始化样本选择器
        Args:
            n_clusters: 聚类数量
            n_init: K-means初始化次数
            max_iter: K-means最大迭代次数
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
    
    def select_samples(self, all_data, all_labels, samples_per_class, feature_extractor):
        """
        从数据中选择样本
        Args:
            all_data: 所有数据
            all_labels: 所有标签
            samples_per_class: 每个类选择的样本数
            feature_extractor: 特征提取器
        Returns:
            selected_indices: 选中的样本索引
            selected_data: 选中的样本数据
            remaining_indices: 剩余样本索引
            remaining_data: 剩余样本数据
        """
        print(f"使用K-means聚类初始化，聚类数: {self.n_clusters}, 每聚类样本数: {samples_per_class}")
        print(f"特征提取方法: {feature_extractor.method}")
        
        X_for_clustering = feature_extractor.extract_features(all_data)
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        
        cluster_labels = kmeans.fit_predict(X_for_clustering)
        distances_all = kmeans.transform(X_for_clustering)
        
        silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
        print(f"聚类完成，轮廓系数: {silhouette_avg:.4f}")
        
        cluster_indices = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in cluster_indices:
                cluster_indices[cluster_id] = []
            cluster_indices[cluster_id].append(idx)
        
        selected_indices = []
        for cluster_id, indices in cluster_indices.items():
            cluster_indices_array = np.array(indices)
            
            if len(indices) >= samples_per_class:
                distances_cluster = distances_all[cluster_indices_array]
                sorted_dist = np.sort(distances_cluster, axis=1)
                dds = sorted_dist[:, 1] - sorted_dist[:, 0]
                
                n_select = min(samples_per_class, len(cluster_indices_array))
                sel_local = np.argsort(dds)[:n_select]
                selected_global_indices = cluster_indices_array[sel_local]
                
                selected_indices.extend(selected_global_indices)
                print(f"聚类 {cluster_id}: 总样本 {len(indices)}, 选择 {n_select} 个 (dds最小-不确定性最大)")
            else:
                selected_indices.extend(indices)
                print(f"警告: 聚类 {cluster_id} 只有 {len(indices)} 个样本，少于要求的 {samples_per_class} 个，全部选择")
        
        selected_indices = np.array(selected_indices)
        
        all_indices = np.arange(len(all_data))
        remaining_indices = np.setdiff1d(all_indices, selected_indices)
        
        selected_data = all_data[selected_indices]
        remaining_data = all_data[remaining_indices]
        
        print(f"使用{feature_extractor.method}特征提取从 {len(cluster_indices)} 个聚类中总计选择了 {len(selected_indices)} 个样本")
        print(f"剩余样本数量: {len(remaining_indices)}")
        
        return selected_indices, selected_data, remaining_indices, remaining_data
    
    @staticmethod
    def create_candidate_pool(remaining_data, remaining_indices, pool_size=100):
        """
        从剩余数据中创建候选池
        Args:
            remaining_data: 剩余数据
            remaining_indices: 剩余数据索引
            pool_size: 候选池大小
        Returns:
            pool_indices: 候选池在原始数据中的索引
            pool_data: 候选池数据
            pool_remaining_indices: 候选池在剩余数据中的索引
        """
        total_remaining = len(remaining_data)
        
        if pool_size > total_remaining:
            print(f"警告: 候选池大小 ({pool_size}) 大于剩余样本数 ({total_remaining})，使用全部剩余样本")
            pool_size = total_remaining
        
        pool_remaining_indices = np.random.choice(total_remaining, pool_size, replace=False)
        
        pool_indices = remaining_indices[pool_remaining_indices]
        
        pool_data = remaining_data[pool_remaining_indices]
        
        return pool_indices, pool_data, pool_remaining_indices

