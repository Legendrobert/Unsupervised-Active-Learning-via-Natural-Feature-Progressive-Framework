# -*- coding: utf-8 -*-
"""
使用SFLM (ELM AutoEncoder)在CIFAR100数据集上进行主动学习
结合ELM和主动学习的优势
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from SFLM.SFLM import ELMAutoEncoder
import time
import torch.optim as optim
import torch.nn.functional as F
import os
import json  
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# 设置随机种子确保结果可复现
def set_seed(seed=42):
    """设置所有随机数生成器的种子"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

# 加载CIFAR100数据集
def load_cifar100():
    """加载CIFAR-100数据集并进行预处理"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])

    train_set = torchvision.datasets.CIFAR100(
        root='/home/lsk/01-Project/Main_data/', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # 转换为NumPy数组并展平
    train_loader = torch.utils.data.DataLoader(
        train_set,  
        batch_size=len(train_set), 
        shuffle=False
    )
    
    # 获取数据和标签
    full_data, full_labels = next(iter(train_loader))
    train_data = full_data.numpy().reshape(len(train_set), -1)  # 展平为[50000, 3072]
    train_labels = full_labels.numpy()
    
    # 加载测试集
    test_set = torchvision.datasets.CIFAR100(
        root='/home/lsk/01-Project/08-Attention_AL/Attention_AL/data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,  
        batch_size=len(test_set), 
        shuffle=False
    )
    
    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.numpy().reshape(len(test_set), -1)  # 展平为[10000, 3072]
    test_labels = test_labels.numpy()
    
    return train_data, train_labels, test_data, test_labels

def get_samples_per_class_from_data(all_data, all_labels, samples_per_class=10, feature_method='resnet'):
    """
    从数据中按类别均匀抽样指定数量的样本
    Args:
        all_data: 所有数据
        all_labels: 所有标签
        samples_per_class: 每个类抽样的数量
        feature_method: 特征提取方法 ('resnet', 'pca', 'tsne', 'raw')
    Returns:
        selected_indices: 选中的样本索引
        selected_data: 选中的样本数据
        remaining_indices: 剩余样本索引
        remaining_data: 剩余样本数据
    """
    # K-means参数设置
    n_clusters = 100  # CIFAR100有100个类别
    samples_per_cluster = samples_per_class  # 每个聚类选择的样本数
    random_state = 42
    
    print(f"使用K-means聚类初始化，聚类数: {n_clusters}, 每聚类样本数: {samples_per_cluster}")
    print(f"特征提取方法: {feature_method}")
    
    # 数据预处理
    if isinstance(all_data, torch.Tensor):
        X_np = all_data.cpu().numpy()
    else:
        X_np = all_data.copy()
    
    # 根据选择的方法进行特征提取
    if feature_method == 'resnet':
        print("使用ResNet18进行特征提取...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练的ResNet18
        import torchvision.models as models
        resnet_model = models.resnet18(pretrained=True)
        # 去掉最后的全连接层，只保留特征提取部分
        resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
        
        # 将数据重新reshape为图像格式并转换为tensor
        # 从 [N, 3072] 转换为 [N, 3, 32, 32]
        X_images = X_np.reshape(-1, 3, 32, 32)
        X_tensor = torch.FloatTensor(X_images).to(device)
        
        # 批量处理以避免内存溢出
        batch_size = 256
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                # 提取特征
                features = resnet_model(batch)
                # 展平特征 [batch_size, 512, 1, 1] -> [batch_size, 512]
                features = features.view(features.size(0), -1)
                features_list.append(features.cpu().numpy())
        
        # 合并所有特征
        X_for_clustering = np.vstack(features_list)
        print(f"ResNet特征提取完成，特征维度: {X_for_clustering.shape}")
        print(f"特征维度从 {X_np.shape[1]} 降至 {X_for_clustering.shape[1]}")
        
    elif feature_method == 'pca':
        print("使用PCA进行特征提取...")
        # PCA参数设置
        n_components = 512  # 保持与ResNet相同的特征维度
        
        # 创建PCA对象
        pca = PCA(n_components=n_components, random_state=random_state)
        
        # 对数据进行PCA降维
        X_for_clustering = pca.fit_transform(X_np)
        
        # 计算解释方差比
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        
        print(f"PCA特征提取完成，特征维度: {X_for_clustering.shape}")
        print(f"特征维度从 {X_np.shape[1]} 降至 {X_for_clustering.shape[1]}")
        print(f"前{n_components}个主成分解释了 {explained_variance_ratio:.4f} ({explained_variance_ratio*100:.2f}%) 的方差")
        
    elif feature_method == 'tsne':
        print("使用t-SNE进行特征提取...")
        from sklearn.manifold import TSNE
        
        # t-SNE参数设置
        n_components = 50  # t-SNE通常用较低维度，50维是一个好的平衡点
        perplexity = 30    # 困惑度，控制关注近邻的数量
        learning_rate = 200  # 学习率
        max_iter = 1000    # 最大迭代次数
        
        # 由于t-SNE计算较慢，对于大数据集先进行PCA降维
        if X_np.shape[1] > 100:
            print("数据维度较高，先使用PCA降至100维以加速t-SNE计算...")
            pca_pre = PCA(n_components=100, random_state=random_state)
            X_pca = pca_pre.fit_transform(X_np)
            pca_var_ratio = np.sum(pca_pre.explained_variance_ratio_)
            print(f"PCA预处理: 保留了 {pca_var_ratio:.4f} ({pca_var_ratio*100:.2f}%) 的方差")
        else:
            X_pca = X_np
        
        # 对于大数据集，使用子集进行t-SNE (可选)
        if len(X_pca) > 10000:
            print(f"数据量较大({len(X_pca)}样本)，为加速计算，使用前10000个样本进行t-SNE...")
            subset_indices = np.random.choice(len(X_pca), 10000, replace=False)
            X_subset = X_pca[subset_indices]
            
            # 在子集上拟合t-SNE
            print("正在计算t-SNE特征...")
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state,
                init='random',
                method='barnes_hut',  # 使用Barnes-Hut近似加速
                verbose=1
            )
            
            X_tsne_subset = tsne.fit_transform(X_subset)
            
            # 对剩余数据，使用k-NN方法映射到t-SNE空间
            print("使用k-NN将剩余数据映射到t-SNE空间...")
            from sklearn.neighbors import NearestNeighbors
            
            # 创建k-NN模型
            k = min(5, len(X_subset))  # 使用5个最近邻
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(X_subset)
            
            # 为所有数据计算t-SNE特征
            X_for_clustering = np.zeros((len(X_pca), n_components))
            X_for_clustering[subset_indices] = X_tsne_subset
            
            # 处理剩余的数据
            remaining_mask = np.ones(len(X_pca), dtype=bool)
            remaining_mask[subset_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) > 0:
                X_remaining = X_pca[remaining_indices]
                distances, indices = knn.kneighbors(X_remaining)
                
                # 使用加权平均来估算t-SNE特征
                weights = 1.0 / (distances + 1e-8)  # 距离的倒数作为权重
                weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化权重
                
                for i, idx in enumerate(remaining_indices):
                    neighbor_features = X_tsne_subset[indices[i]]
                    X_for_clustering[idx] = np.average(neighbor_features, weights=weights[i], axis=0)
        else:
            # 对于小数据集，直接使用t-SNE
            print("正在计算t-SNE特征...")
            tsne = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, (len(X_pca) - 1) // 3),  # 调整困惑度以适应数据量
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state,
                init='random',
                method='barnes_hut' if len(X_pca) > 1000 else 'exact',
                verbose=1
            )
            
            X_for_clustering = tsne.fit_transform(X_pca)
        
        print(f"t-SNE特征提取完成，特征维度: {X_for_clustering.shape}")
        print(f"特征维度从 {X_np.shape[1]} 降至 {X_for_clustering.shape[1]}")
        print(f"t-SNE参数: perplexity={perplexity}, n_components={n_components}")
        
    elif feature_method == 'raw':
        print("使用原始数据进行聚类...")
        X_for_clustering = X_np
        print(f"使用原始特征维度: {X_for_clustering.shape}")
        
    else:
        raise ValueError(f"不支持的特征提取方法: {feature_method}. 支持的方法: 'resnet', 'pca', 'tsne', 'raw'")
    
    # 执行K-means聚类
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(X_for_clustering)
    cluster_centers = kmeans.cluster_centers_
    # 新增：计算每个样本到所有中心的距离矩阵
    distances_all = kmeans.transform(X_for_clustering)
     
    # 计算聚类质量
    silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
    print(f"聚类完成，轮廓系数: {silhouette_avg:.4f}")
    
    # 按聚类分组
    cluster_indices = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_indices:
            cluster_indices[cluster_id] = []
        cluster_indices[cluster_id].append(idx)
    
    selected_indices = []
    for cluster_id, indices in cluster_indices.items():
        cluster_indices_array = np.array(indices)
        
        if len(indices) >= samples_per_cluster:
            # 使用 dds 选择不确定性最大的样本（次近-最近距离差最小）
            distances_cluster = distances_all[cluster_indices_array]
            sorted_dist = np.sort(distances_cluster, axis=1)
            dds = sorted_dist[:, 1] - sorted_dist[:, 0]
            
            n_select = min(samples_per_cluster, len(cluster_indices_array))
            sel_local = np.argsort(dds)[:n_select]
            selected_global_indices = cluster_indices_array[sel_local]
            
            selected_indices.extend(selected_global_indices)
            print(f"聚类 {cluster_id}: 总样本 {len(indices)}, 选择 {n_select} 个 (dds最小-不确定性最大)")
        else:
            selected_indices.extend(indices)  # 如果样本不足，全部选择
            print(f"警告: 聚类 {cluster_id} 只有 {len(indices)} 个样本，少于要求的 {samples_per_cluster} 个，全部选择")
    
    selected_indices = np.array(selected_indices)
    
    # 计算剩余样本索引
    all_indices = np.arange(len(all_data))
    remaining_indices = np.setdiff1d(all_indices, selected_indices)
    
    # 提取数据
    selected_data = all_data[selected_indices]
    remaining_data = all_data[remaining_indices]
    
    print(f"使用{feature_method}特征提取从 {len(cluster_indices)} 个聚类中总计选择了 {len(selected_indices)} 个样本")
    print(f"剩余样本数量: {len(remaining_indices)}")

    return selected_indices, selected_data, remaining_indices, remaining_data

def create_candidate_pool_from_remaining(remaining_data, remaining_indices, pool_size=100):
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
    
    # 在剩余数据中随机选择
    pool_remaining_indices = np.random.choice(total_remaining, pool_size, replace=False)
    
    # 获取在原始数据中的索引
    pool_indices = remaining_indices[pool_remaining_indices]
    
    # 获取候选池数据
    pool_data = remaining_data[pool_remaining_indices]
    
    return pool_indices, pool_data, pool_remaining_indices

def get_labels_from_data(all_labels, indices):
    """
    从标签数组中获取指定索引的标签
    Args:
        all_labels: 所有标签
        indices: 样本索引数组
    Returns:
        labels: 对应的标签数组
    """
    return all_labels[indices]

def print_class_distribution(data_indices, all_labels, title="数据集"):
    """
    打印数据集的类别分布
    Args:
        data_indices: 数据索引
        all_labels: 所有标签
        title: 标题
    """
    labels = get_labels_from_data(all_labels, data_indices)
    class_counts = {}
    
    for label in labels:
        class_num = int(label)  # 直接使用数字类别
        class_counts[class_num] = class_counts.get(class_num, 0) + 1
    
    print(f"\n{title}类别分布 (总计 {len(data_indices)} 个样本):")
    print("-" * 50)
    
    # 只显示有样本的类别，按类别编号排序
    for class_num in sorted(class_counts.keys()):
        count = class_counts[class_num]
        percentage = (count / len(data_indices)) * 100 if len(data_indices) > 0 else 0
        print(f"类别{class_num:<3}: {count:>4} 个样本 ({percentage:>5.1f}%)")

def train_elm_model_simple(model, data, model_name):
    """
    简化的ELM模型训练函数，不进行可视化
    """
    print(f"正在训练 {model_name}...")
    start_time = time.time()
    model.fit(data)
    training_time = time.time() - start_time
    
    # 计算训练完成后的重构误差
    reconstructed = model.predict(data)
    sample_mse = np.mean((data - reconstructed) ** 2, axis=1)
    final_mse = np.mean((data - reconstructed) ** 2)
    
    print(f"{model_name} 训练完成，耗时: {training_time:.2f}秒，MSE: {final_mse:.6f}")
    return final_mse, sample_mse

def compute_elm_loss_differences(ref_sample_mse, current_model, pool_data, pool_indices, loss_type='mse'):
    """
    计算ELM模型的损失差值，支持MSE和皮尔逊相关系数
    Args:
        ref_sample_mse: 参考模型在所有训练数据上的每个样本MSE
        current_model: 当前ELM模型
        pool_data: 候选池数据 (torch tensor)
        pool_indices: 候选池索引
        loss_type: 损失类型 ('mse', 'pearson')
    Returns:
        loss_differences: 损失差值数组，值越大表示当前模型重构质量越差
        ref_losses: 参考模型的损失数组
        current_losses: 当前模型的损失数组
    """
    print(f"正在计算ELM {loss_type.upper()}差值...")
    
    # 将torch tensor转换为numpy array
    if isinstance(pool_data, torch.Tensor):
        pool_data_np = pool_data.cpu().numpy()
    else:
        pool_data_np = pool_data
    
    if loss_type.lower() == 'mse':
        # 从预计算的ref_sample_mse中获取参考模型的MSE
        ref_losses = ref_sample_mse[pool_indices]
        
        # 计算当前模型在候选池数据上的MSE
        current_reconstructed = current_model.predict(pool_data_np)
        current_losses = np.mean((pool_data_np - current_reconstructed) ** 2, axis=1)

        # 计算MSE差值 (current_mse - ref_mse)
        # MSE值越低表示质量越好，所以差值为正表示当前模型更差
        # 值越大表示当前模型越差
        loss_differences = current_losses - ref_losses
        
    elif loss_type.lower() == 'pearson':
        # 计算当前模型的重构结果
        current_reconstructed = current_model.predict(pool_data_np)
        
        # 为皮尔逊相关系数计算准备数据
        num_samples = len(pool_data_np)
        pearson_scores = np.zeros(num_samples)
        
        # 对每个样本计算皮尔逊相关系数
        for i in range(num_samples):
            original = pool_data_np[i].flatten()
            reconstructed = current_reconstructed[i].flatten()
            
            # 计算皮尔逊相关系数
            try:
                correlation, _ = pearsonr(original, reconstructed)
                # 处理可能的NaN值
                if np.isnan(correlation):
                    correlation = 0.0
                pearson_scores[i] = correlation
            except:
                # 如果计算失败，设为0
                pearson_scores[i] = 0.0
        
        # 皮尔逊系数越高表示相关性越强，重构质量越好
        # 我们需要的是损失值（越大表示越差），所以使用1-pearson
        current_losses = 1.0 - pearson_scores
        
        # 从参考模型MSE构造参考损失（这里我们假设MSE可以转换为相对的质量指标）
        # 注意：这里是一个近似，因为MSE和皮尔逊系数衡量的是不同的方面
        ref_mse_normalized = ref_sample_mse[pool_indices]
        # 将MSE标准化到[0,1]范围作为参考损失
        if ref_mse_normalized.max() > ref_mse_normalized.min():
            ref_losses = (ref_mse_normalized - ref_mse_normalized.min()) / (ref_mse_normalized.max() - ref_mse_normalized.min())
        else:
            ref_losses = np.zeros_like(ref_mse_normalized)
        
        # 计算损失差值
        loss_differences = current_losses - ref_losses
        
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}. 支持的类型: 'mse', 'pearson'")
    
    print(f"ELM {loss_type.upper()}差值计算完成，共处理 {len(pool_data_np)} 个样本")
    
    if loss_type.lower() == 'pearson':
        print(f"参考损失范围: [{ref_losses.min():.6f}, {ref_losses.max():.6f}]")
        print(f"当前模型Pearson损失范围: [{current_losses.min():.6f}, {current_losses.max():.6f}]")
        print(f"Pearson损失差值范围: [{loss_differences.min():.6f}, {loss_differences.max():.6f}]")
        print(f"原始Pearson系数范围: [{(1.0-current_losses).min():.6f}, {(1.0-current_losses).max():.6f}]")
    else:
        print(f"参考模型MSE范围: [{ref_losses.min():.6f}, {ref_losses.max():.6f}]")
        print(f"当前模型MSE范围: [{current_losses.min():.6f}, {current_losses.max():.6f}]")
        print(f"MSE差值范围: [{loss_differences.min():.6f}, {loss_differences.max():.6f}]")
    
    return loss_differences, ref_losses, current_losses

def print_elm_loss_results(pool_indices, loss_differences, ref_losses, current_losses, pool_labels, loss_type='mse', top_k=10):
    """
    打印ELM损失差值结果（包含类别标签）
    Args:
        pool_indices: 候选池索引
        loss_differences: 损失差值数组
        ref_losses: 参考模型损失数组
        current_losses: 当前模型损失数组
        pool_labels: 候选池标签
        loss_type: 损失类型 ('mse', 'pearson')
        top_k: 显示前k个结果
    """
    # 创建结果字典
    results = []
    for i, (idx, diff, ref_loss, curr_loss, label) in enumerate(zip(pool_indices, loss_differences, ref_losses, current_losses, pool_labels)):
        results.append({
            'index': idx,
            'difference': diff,
            'ref_loss': ref_loss,
            'current_loss': curr_loss,
            'label': label,
            'class_num': int(label)  # 直接使用数字类别
        })
    
    # 按差值降序排序
    results.sort(key=lambda x: x['difference'], reverse=True)
    
    loss_name = loss_type.upper()
    print(f"\nTop {top_k} Largest {loss_name} Differences ({loss_name}_current - {loss_name}_ref):")
    print("Index\t\tClass\t\t{}_ref\t\t{}_current\t\tDifference".format(loss_name, loss_name))
    
    print("-" * 80)
    for i in range(min(top_k, len(results))):
        result = results[i]
        print(f"{result['index']}\t\t{result['class_num']:<6}\t\t{result['ref_loss']:.8f}\t{result['current_loss']:.8f}\t{result['difference']:.8f}")
    
    # 统计信息
    print(f"\nELM {loss_name} Difference Statistics:")
    print(f"Mean difference: {np.mean(loss_differences):.6f}")
    print(f"Std difference: {np.std(loss_differences):.6f}")
    print(f"Max difference: {np.max(loss_differences):.6f}")
    print(f"Min difference: {np.min(loss_differences):.6f}")
    
    print(f"Positive differences (current > ref): {np.sum(loss_differences > 0)}/{len(loss_differences)}")
    print(f"Negative differences (current < ref): {np.sum(loss_differences < 0)}/{len(loss_differences)}")
    
    # 按类别统计
    print(f"\nClass Distribution in Top {top_k}:")
    class_counts = {}
    for i in range(min(top_k, len(results))):
        class_num = results[i]['class_num']
        class_counts[class_num] = class_counts.get(class_num, 0) + 1
    
    for class_num, count in sorted(class_counts.items()):
        print(f"类别{class_num}: {count}")

def visualize_freqaware_results(pool_data, pool_indices, pool_labels, ref_model, current_model, top_k=5):
    """
    可视化FreqAware模型的重建结果对比
    """
    # 获取前k个样本
    sample_indices = np.random.choice(len(pool_data), min(top_k, len(pool_data)), replace=False)
    
    if isinstance(pool_data, torch.Tensor):
        sample_data = pool_data[sample_indices].numpy()
    else:
        sample_data = pool_data[sample_indices]
    
    # 重建
    ref_reconstructed = ref_model.predict(sample_data)
    current_reconstructed = current_model.predict(sample_data)
    
    plt.figure(figsize=(20, 12))
    plt.suptitle('FreqAware Model Reconstruction Comparison', fontsize=16)
    
    for i in range(len(sample_indices)):
        idx = sample_indices[i]
        global_idx = pool_indices[idx]
        class_num = int(pool_labels[idx])  # 直接使用数字类别
        
        # 原图
        plt.subplot(4, len(sample_indices), i+1)
        img = sample_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = img * 0.5 + 0.5  # 反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Original\nIndex: {global_idx} (Class {class_num})")
        plt.axis('off')
        
        # 参考模型重建
        plt.subplot(4, len(sample_indices), i+1+len(sample_indices))
        img = ref_reconstructed[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = img * 0.5 + 0.5  # 反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title("Ref Model\nReconstruction")
        plt.axis('off')
        
        # 当前模型重建
        plt.subplot(4, len(sample_indices), i+1+2*len(sample_indices))
        img = current_reconstructed[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = img * 0.5 + 0.5  # 反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title("Current Model\nReconstruction")
        plt.axis('off')
        
        # 差异图
        plt.subplot(4, len(sample_indices), i+1+3*len(sample_indices))
        diff = np.abs(sample_data[i] - current_reconstructed[i])
        diff_img = diff.reshape(3, 32, 32).transpose(1, 2, 0)
        diff_img = diff_img / diff_img.max() if diff_img.max() > 0 else diff_img
        plt.imshow(diff_img)
        plt.title("Error Map")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 设置随机种子
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("正在加载CIFAR100数据集...") 
    
    # 加载CIFAR100数据集
    all_train_data, all_train_labels, test_data, test_labels = load_cifar100()
    
    input_dim = 3072  # CIFAR100图像展平后的维度 (3*32*32)
    # target_size = int(0.4*len(all_train_data))  # 目标数据量
    target_size = 1000
    
    print(f"训练数据形状: {all_train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    print(f"目标数据量: {target_size} 样本")
    print(f"CIFAR100总类别数: 100")
    
    # 创建并训练参考ELM模型（ref_model）- 只训练一次
    print("="*60)
    print("创建并训练参考ELM模型...")
    ref_model = ELMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=1800,        # 减小隐藏层避免内存问题
        activation='sigmoid',
        C=1e-2,
        epochs=1,
        type_=1,
        seed=42
    )
    

    # # 使用较小的数据集进行初始训练以避免内存问题
    # train_subset_size = 5000
    # train_subset = all_train_data[:train_subset_size]
    # print(f"使用 {train_subset_size} 个样本训练参考模型以节省内存")
    
    ref_final_mse, ref_final_sample_mse = train_elm_model_simple(ref_model, all_train_data, "参考ELM模型")

    # 添加参考模型样本MSE统计
    print(f"参考模型每个样本MSE统计:")
    print(f"  样本数量: {len(ref_final_sample_mse)}")
    print(f"  最小值: {ref_final_sample_mse.min():.6f}")
    print(f"  最大值: {ref_final_sample_mse.max():.6f}")
    print(f"  均值: {ref_final_sample_mse.mean():.6f}")
    print(f"  标准差: {ref_final_sample_mse.std():.6f}")
    print(f"  中位数: {np.median(ref_final_sample_mse):.6f}")
    
    # 初始化：从全部训练数据中均匀抽样
    print("="*60)
    print("初始化：从全部训练数据中均匀抽样样本...")
    
    # 可以选择不同的特征提取方法: 'resnet', 'pca', 'raw'
    feature_method = 'resnet'  # 使用PCA特征提取
    
    # 可以选择不同的损失计算方法: 'mse', 'pearson'
    loss_type = 'pearson'  # 使用Pearson相关系数损失计算

    print(f"特征提取方法: {feature_method}")
    print(f"损失计算方法: {loss_type.upper()}")
    
    current_indices, current_data, remaining_indices, remaining_data = get_samples_per_class_from_data(
        all_train_data, all_train_labels, samples_per_class=int(0.3*target_size*0.01), feature_method=feature_method)  # 减少初始样本数
    
    print_class_distribution(current_indices, all_train_labels, "初始")
    
    # 主动学习迭代过程
    print("="*60)
    print("开始ELM主动学习迭代...")
          
    # 计算预估迭代次数
    samples_per_iteration = 10  # 减少每次迭代的样本数
    estimated_iterations = (target_size - len(current_data)) // samples_per_iteration
    
    # 使用tqdm显示进度
    pbar = tqdm(total=target_size - len(current_data), 
                desc="ELM主动学习进度", 
                unit="样本")
    
    iteration = 0
    
    while len(current_data) < target_size:
        iteration += 1
        
        # 创建并训练当前ELM模型
        current_model = ELMAutoEncoder(
            input_dim=input_dim,
            hidden_dim=10+iteration,         #
            activation='sine',  
            C=1e-2,
            epochs=1,
            type_=0,
            seed=42
        )
        
        # 静默训练（不打印训练信息）
        current_model.fit(current_data)
        
        # 从剩余样本中创建候选池
        if len(remaining_data) == 0:
            break
            
        pool_size = min(1500, len(remaining_data))  # 减小候选池大小
        pool_indices, pool_data, pool_remaining_indices = create_candidate_pool_from_remaining(
            remaining_data, remaining_indices, pool_size=pool_size)
        
        # 计算ELM损失差值（使用选择的损失类型）
        pool_data_tensor = torch.from_numpy(pool_data) if isinstance(pool_data, np.ndarray) else pool_data
        loss_differences, ref_losses, current_losses = compute_elm_loss_differences(
            ref_final_sample_mse, current_model, pool_data_tensor, pool_indices, loss_type=loss_type)
        
        # 选择损失差值最大的前几个样本
        num_to_select = min(5, len(remaining_data), target_size - len(current_data))  # 减少选择数量
        
        top_diff_indices = np.argsort(loss_differences)[-num_to_select:][::-1]
        # top_diff_indices = np.argsort(loss_differences)[-num_to_select:]
        selected_pool_indices = pool_remaining_indices[top_diff_indices]
        selected_original_indices = remaining_indices[selected_pool_indices]
        
        # 将选中的样本添加到current_data中
        selected_data = remaining_data[selected_pool_indices]
        current_data = np.vstack([current_data, selected_data])
        current_indices = np.concatenate([current_indices, selected_original_indices])
        
        # 从剩余数据中移除已选择的样本
        remaining_mask = np.ones(len(remaining_data), dtype=bool)
        remaining_mask[selected_pool_indices] = False
        remaining_data = remaining_data[remaining_mask]
        remaining_indices = remaining_indices[remaining_mask]
        
        # 打印选中样本的类别分布
        selected_labels = get_labels_from_data(all_train_labels, selected_original_indices)
        class_counts = {}
        for label in selected_labels:
            class_num = int(label)  # 直接使用数字类别
            class_counts[class_num] = class_counts.get(class_num, 0) + 1
        
        # 格式化类别分布信息（只显示有样本的类别）
        class_info = ", ".join([f"{cls}:{cnt}" for cls, cnt in sorted(class_counts.items()) if cnt > 0])
        
        # 更新进度条
        pbar.update(num_to_select)
        pbar.set_postfix({
            'iter': iteration,
            'current': len(current_data),
            'selected': class_info,
            'loss': loss_type.upper()
        })
        
        # 每隔几次迭代显示详细结果
        if iteration % 10 == 0:  # 减少打印频率
            pool_labels = get_labels_from_data(all_train_labels, pool_indices)
            print_elm_loss_results(
                pool_indices, loss_differences, ref_losses, current_losses, 
                pool_labels, loss_type=loss_type, top_k=3  # 减少显示数量
            )
    
    pbar.close()
    
    # 最终训练和评估
    print("\n" + "="*60)
    print("ELM主动学习完成，进行最终评估...")
    
    # 最终训练ELM模型
    final_current_model = ELMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=1000,
        activation='sigmoid',
        C=1e-2,
        epochs=1,
        type_=1,
        seed=42
    )
    
    final_current_mse, final_current_sample_mse = train_elm_model_simple(
        final_current_model, current_data, "最终ELM模型"
    )
    
    # 在测试集上评估（使用较小子集）
    print("在测试集上评估ELM模型...")
    test_subset = test_data[:200]  # 进一步减少测试集大小
    
    ref_test_reconstructed = ref_model.predict(test_subset)
    final_current_test_reconstructed = final_current_model.predict(test_subset)
    
    ref_test_mse = np.mean((test_subset - ref_test_reconstructed) ** 2)
    final_current_test_mse = np.mean((test_subset - final_current_test_reconstructed) ** 2)
    
    # 打印最终结果和类别分布
    print("="*60)
    print("ELM主动学习最终结果总结:")
    # print(f"参考模型训练数据量: {len(train_subset)} 样本")
    print(f"最终当前模型训练数据量: {len(current_data)} 样本")
    print(f"参考模型训练集 - MSE: {ref_final_mse:.6f}")
    print(f"最终当前模型训练集 - MSE: {final_current_mse:.6f}")
    print(f"参考模型测试集 - MSE: {ref_test_mse:.6f}")
    print(f"最终当前模型测试集 - MSE: {final_current_test_mse:.6f}")
    print(f"总迭代次数: {iteration}")
    print(f"目标达成: {len(current_data) >= target_size}")
    print(f"特征提取方法: {feature_method}")
    print(f"损失计算方法: {loss_type.upper()}")
    
    # 打印最终选择的数据类别分布
    print_class_distribution(current_indices, all_train_labels, "最终选择的")
    
    # 保存索引和结果
    np.save('Results/elm_selected_indices_cifar100_1000.npy', current_indices)
    
    # 保存详细结果
    results = {
        'ref_model_results': {
            'train_mse': float(ref_final_mse),  # 转换为Python float
            'test_mse': float(ref_test_mse),
            'data_size': int(len(all_train_data))  # 使用完整训练集大小，转换为Python int
        },
        'final_model_results': {
            'train_mse': float(final_current_mse),
            'test_mse': float(final_current_test_mse),
            'data_size': int(len(current_data))
        },
        'experiment_config': {
            'target_size': int(target_size),
            'feature_method': feature_method,
            'loss_type': loss_type,
            'iterations': int(iteration),
            'samples_per_iteration': int(samples_per_iteration)
        }
    }
    
    import json
    with open('SFLM/elm_al_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n已保存选择的样本索引到 'SFLM/elm_selected_indices_cifar100_1000.npy'")
    print(f"已保存实验结果到 'SFLM/elm_al_results.json'")
    print(f"保存的索引数量: {len(current_indices)}")
    
    print("ELM主动学习测试完成！")

if __name__ == "__main__":
    main() 