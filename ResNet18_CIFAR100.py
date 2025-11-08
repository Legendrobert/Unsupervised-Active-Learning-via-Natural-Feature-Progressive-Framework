import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from sklearn.metrics import confusion_matrix
import random
import os  # 添加os模块

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01

# CIFAR-100类别名称（英文）
CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# 训练函数
def train(model, trainloader, optimizer, criterion, epoch, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/len(trainloader), 100.*correct/total

# 测试函数
def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # 用于计算混淆矩阵
    all_targets = []
    all_predictions = []
    
    # 各类别正确预测和总数
    class_correct = [0] * 100  # 修改为100个类别
    class_total = [0] * 100    # 修改为100个类别
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 收集所有预测和目标用于混淆矩阵
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # 计算每个类别的准确率
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # 计算每个类别的准确率
    class_accuracies = [100 * class_correct[i] / max(1, class_total[i]) for i in range(100)]  # 修改为100个类别
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    return test_loss/len(testloader), 100.*correct/total, conf_matrix, class_accuracies

def create_balanced_subset(dataset, percentage=0.4):
    """
    创建一个平衡的数据子集，从每个类别中随机选择相同比例的样本
    
    Args:
        dataset: 完整的数据集
        percentage: 要选择的样本比例 (0-1)
        
    Returns:
        所选样本的索引列表
    """
    # 获取所有标签
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # 创建类别索引字典
    class_indices = {}
    for i in range(100):  # CIFAR-100有100个类别
        class_indices[i] = np.where(targets == i)[0]
    
    # 从每个类别随机选择指定比例的样本
    selected_indices = []
    for class_id, indices in class_indices.items():
        num_to_select = int(len(indices) * percentage)
        selected = np.random.choice(indices, num_to_select, replace=False)
        selected_indices.extend(selected)
    
    # 打乱索引顺序
    np.random.shuffle(selected_indices)
    
    return selected_indices

def main():
    set_seed(42)
    print(f"使用设备: {DEVICE}")
    
    # 创建保存结果的目录
    results_dir = "results/resnet_epoch"
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到: {results_dir}")
    
    # 数据变换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. 加载数据集
    print("准备数据集...")
    # 加载完整的CIFAR-100数据集
    trainset_full = torchvision.datasets.CIFAR100(  # 修改为CIFAR100
        root='/home/lsk/01-Project/Main_data/', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(  # 修改为CIFAR100
        root='/home/lsk/01-Project/Main_data/', train=False, download=True, transform=transform_test)

    # 1. 选择数据子集方式
    subset_mode = "file_npy"  # 可以更改为: "file_npy", "file_json", "random_balanced"
    
    if subset_mode == "file_npy":
        # 方法1：直接加载NPY文件 
        print("加载NPY索引文件...")
        try:
            # selected_indices = np.load('SFLM/freqaware_selected_indices_cifar100_1000.npy')
            selected_indices = np.load('./Results/elm_selected_indices_cifar100_1000.npy')
            # selected_indices = np.load('results/paper_selected_subset_indices_cifar100_subset4500.npy')
            print("成功加载NPY文件")
        except FileNotFoundError:
            print("错误：无法找到NPY索引文件。")
            exit(1)
    elif subset_mode == "file_json":
        # 方法2：从JSON文件加载
        print("加载JSON索引文件...")
        try:
            with open('subset_selection_results.json', 'r') as f:
                subset_data = json.load(f)
                selected_indices = np.array(subset_data['indices'])
            print("成功加载JSON文件")
        except FileNotFoundError:
            print("错误：无法找到JSON索引文件。")
            exit(1)
    elif subset_mode == "random_balanced":
        # 方法3：随机选择40%的样本，每个类别平均分布
        print("创建随机平衡子集（每个类别选择40%的样本）...")
        selected_indices = create_balanced_subset(trainset_full, percentage=0.4)
        print("成功创建随机平衡子集")
    else:
        print("错误：无效的子集选择模式")
        exit(1)

    print(f"成功获取{len(selected_indices)}个样本索引")

    # 使用选定的索引创建训练子集
    trainset = Subset(trainset_full, selected_indices)

    # 创建数据加载器 - 增加num_workers以加速数据加载
    num_workers = 8  # 可以设置为CPU核心数的2-4倍
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,                                                 
        num_workers=num_workers, pin_memory=True)  # 添加pin_memory=True
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=num_workers, pin_memory=True)
  
    print(f"训练集: {len(trainset)}个样本")
    print(f"测试集: {len(testset)}个样本")

    # 3. 创建ResNet18模型
    print("初始化ResNet18模型...")
    # model = resnet18(weights=None)
    model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 100)  # 修改输出为100个类别
    )
    model = model.to(DEVICE)

    # 4. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 7. 训练和测试模型
    print("开始训练...")
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # 添加最佳准确率跟踪
    best_acc = 0
    best_epoch = 0
    
    # 创建训练历史记录
    training_history = {
        'epochs': [],
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, epoch, DEVICE)
        test_loss, test_acc, _, _ = test(model, testloader, criterion, DEVICE)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存到训练历史记录
        training_history['epochs'].append(epoch + 1)
        training_history['train_losses'].append(float(train_loss))
        training_history['train_accs'].append(float(train_acc))
        training_history['test_losses'].append(float(test_loss))
        training_history['test_accs'].append(float(test_acc))
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            # 保存最佳模型
            torch.save(model.state_dict(), 'resnet18_subset_best.pth')
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")
    
    # 保存完整的训练历史到单个文件
    training_results = {
        'epochs': training_history['epochs'],
        'train_losses': training_history['train_losses'],
        'train_accs': training_history['train_accs'],
        'test_losses': training_history['test_losses'],
        'test_accs': training_history['test_accs'],
        'best_accuracy': float(best_acc),
        'best_epoch': best_epoch,
        'final_accuracy': float(test_accs[-1]),
        'training_epochs': EPOCHS
    }
    
    # 保存为单个JSON文件
    results_file = os.path.join(results_dir, 'training_results_paper.json')
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=4)
    
    print(f"所有训练结果已保存到: {results_file}")

    # 8. 保存模型
    print("Saving model...")
    torch.save(model.state_dict(), 'resnet18_subset_final.pth')
    
    # 绘制训练历史
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # 计算最终的混淆矩阵和各类别准确率
    print("Computing confusion matrix and per-class accuracy...")  
    _, final_acc, conf_matrix, class_accuracies = test(model, testloader, criterion, DEVICE)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12))  # 增大图像尺寸以适应100个类别
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',  # 关闭数字标注以避免过于拥挤
                xticklabels=False, yticklabels=False)  # 关闭标签以避免过于拥挤
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (CIFAR-100)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)  # 增加分辨率
    
    # 绘制各类别准确率
    plt.figure(figsize=(20, 8))  # 增大图像宽度以适应100个类别
    bars = plt.bar(range(100), class_accuracies)  # 修改为100个类别
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy (CIFAR-100)')
    plt.xticks(range(0, 100, 10), [f'{i}' for i in range(0, 100, 10)])  # 只显示每10个类别的标签
    
    # 添加平均准确率线
    avg_acc = np.mean(class_accuracies)
    plt.axhline(y=avg_acc, color='red', linestyle='--', label=f'Average: {avg_acc:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300)  # 增加分辨率
    
    # 保存最佳准确率信息
    best_model_info = {
        'best_accuracy': float(best_acc),
        'best_epoch': best_epoch,
        'final_accuracy': float(test_accs[-1]),
        'training_epochs': EPOCHS,
        'per_class_accuracy': {CLASSES[i]: float(class_accuracies[i]) for i in range(100)},  # 修改为100个类别
        'average_per_class_accuracy': float(np.mean(class_accuracies))  # 添加平均准确率
    }
    
    with open('best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=4)

    print(f"Training completed! Final test accuracy: {test_accs[-1]:.2f}%")
    print(f"Best test accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"Training history and model saved. Best model saved as 'resnet18_subset_best.pth'")
    print(f"Confusion matrix and per-class accuracy saved as images")

if __name__ == "__main__":
    # Windows多进程支持
    multiprocessing.freeze_support()
    main()
