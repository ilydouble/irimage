
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b3, EfficientNet_B3_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
# from ultralytics import YOLO  # 删除这行
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 计算alpha权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ThermalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class ThermalClassifier:
    def __init__(self, data_dir="./dataset/datasets/thermal_classification_cropped",
                 output_dir="./results/training_results/thermal_classifier_results"):
        self.data_dir = Path(data_dir)
        self.base_output_dir = Path(output_dir)
        # 输出目录将在run_training中根据参数创建
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        print(f"基础输出目录: {self.base_output_dir}")
        
        # 类别映射
        self.class_to_idx = {'non_icas': 0, 'icas': 1}
        self.idx_to_class = {0: 'non_icas', 1: 'icas'}
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),  # 增加到512x512
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # 添加高斯模糊
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),  # 验证时也用512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_descriptive_output_dir(self, backbone, batch_size, num_epochs, lr, use_focal_loss):
        """创建描述性的输出目录"""
        from datetime import datetime

        # 构建描述性文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss_type = "focal" if use_focal_loss else "ce"  # ce = cross entropy

        # 格式: CNN-ResNet18-Focal-E100-LR0.0001-B32-20240829_143022
        folder_name = f"CNN-{backbone.upper()}-{loss_type.upper()}-E{num_epochs}-LR{lr}-B{batch_size}-{timestamp}"

        output_dir = self.base_output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir
    
    def load_data(self):
        """加载数据并划分数据集"""
        print("加载数据...")
        
        image_paths = []
        labels = []
        
        # 遍历每个类别目录
        for class_name in ['icas', 'non_icas']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: 类别目录不存在 {class_dir}")
                continue
            
            class_label = self.class_to_idx[class_name]
            
            # 收集该类别的所有图像
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(class_label)
        
        print(f"总共找到 {len(image_paths)} 张图像")
        print(f"ICAS: {labels.count(1)} 张")
        print(f"Non-ICAS: {labels.count(0)} 张")
        
        # 划分数据集: 70% 训练, 15% 验证, 15% 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        print(f"\n数据集划分:")
        print(f"训练集: {len(X_train)} 张 (ICAS: {y_train.count(1)}, Non-ICAS: {y_train.count(0)})")
        print(f"验证集: {len(X_val)} 张 (ICAS: {y_val.count(1)}, Non-ICAS: {y_val.count(0)})")
        print(f"测试集: {len(X_test)} 张 (ICAS: {y_test.count(1)}, Non-ICAS: {y_test.count(0)})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self, backbone='resnet18'):
        """创建分类模型"""
        print(f"创建模型: {backbone}")
        
        if backbone.startswith('yolo11'):
            # 只在需要时导入
            from ultralytics import YOLO
            
            if backbone == 'yolo11n':
                yolo_model = YOLO('yolo11n.pt')
            elif backbone == 'yolo11s':
                yolo_model = YOLO('yolo11s.pt')
            elif backbone == 'yolo11m':
                yolo_model = YOLO('yolo11m.pt')
            elif backbone == 'yolo11l':
                yolo_model = YOLO('yolo11l.pt')
            elif backbone == 'yolo11x':
                yolo_model = YOLO('yolo11x.pt')
            else:
                raise ValueError(f"不支持的YOLO11模型: {backbone}")
            
            # 提取YOLO11的骨干网络
            backbone_model = yolo_model.model.model[:10]  # 取前10层作为特征提取器
            
            # 创建分类头
            class YOLO11Classifier(nn.Module):
                def __init__(self, backbone, backbone_name, num_classes=2):
                    super().__init__()
                    self.backbone = backbone
                    self.backbone_name = backbone_name
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    
                    # 动态获取特征维度
                    self.feature_dim = self._get_feature_dim()
                    
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(self.feature_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, num_classes)
                    )
                
                def _get_feature_dim(self):
                    """动态获取特征维度"""
                    # 创建一个测试输入
                    test_input = torch.randn(1, 3, 512, 512)
                    with torch.no_grad():
                        x = test_input
                        for layer in self.backbone:
                            x = layer(x)
                        x = self.avgpool(x)
                        x = torch.flatten(x, 1)
                        return x.shape[1]
                
                def forward(self, x):
                    # 通过YOLO11骨干网络
                    for layer in self.backbone:
                        x = layer(x)
                    
                    # 全局平均池化
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    
                    # 分类
                    x = self.classifier(x)
                    return x
            
            model = YOLO11Classifier(backbone_model, backbone, num_classes=2)
            num_features = f"YOLO11特征提取器 (dim: {model.feature_dim})"

        elif backbone == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
        elif backbone == 'resnet34':
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
        elif backbone == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
        elif backbone == 'efficientnet_b0':
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Linear(num_features, 2)
            
        elif backbone == 'efficientnet_b1':
            model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Linear(num_features, 2)
            
        elif backbone == 'efficientnet_b2':
            model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Linear(num_features, 2)
            
        elif backbone == 'efficientnet_b3':
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Linear(num_features, 2)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        print(f"模型已加载，特征维度: {num_features}")
        return model.to(self.device)
    
    def create_balanced_weights(self, y_train):
        """计算类别权重"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        print(f"类别权重:")
        print(f"  Non-ICAS (0): {class_weights[0]:.3f}")
        print(f"  ICAS (1): {class_weights[1]:.3f}")
        
        return torch.FloatTensor(class_weights).to(self.device)

    def train_model(self, model, train_loader, val_loader, num_epochs=50, lr=0.001, 
                   weight_decay=1e-4, patience=10, class_weights=None, use_focal_loss=True):
        """训练模型 - 添加Focal Loss支持"""
        print(f"开始训练，共 {num_epochs} 个epoch...")
        
        # 选择损失函数
        if use_focal_loss:
            # 使用Focal Loss
            if class_weights is not None:
                criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
            else:
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print("使用Focal Loss损失函数")
        else:
            # 使用加权交叉熵损失
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print("使用加权交叉熵损失函数")
            else:
                criterion = nn.CrossEntropyLoss()
                print("使用标准交叉熵损失函数")
        
        # 使用更小的学习率和更强的正则化
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        # 早停机制 - 更严格的条件
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        learning_rates = []
        
        # 添加更多指标跟踪
        train_precisions = []
        train_recalls = []
        val_precisions = []
        val_recalls = []
        
        best_val_f1 = 0.0  # 使用F1分数而不是准确率
        best_model_path = self.output_dir / "best_model.pth"
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_predictions = []
            train_targets = []
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for images, labels in train_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 收集预测结果用于计算精确率和召回率
                train_predictions.extend(predicted.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
                
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'LR': f'{current_lr:.6f}'
                })
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            # 计算指标
            from sklearn.metrics import precision_recall_fscore_support, f1_score
            
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                train_targets, train_predictions, average='weighted', zero_division=0
            )
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_targets, val_predictions, average='weighted', zero_division=0
            )
            
            # 计算ICAS类别的F1分数（更重要的指标）
            icas_f1 = f1_score(val_targets, val_predictions, pos_label=1, zero_division=0)
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, P: {train_precision:.3f}, R: {train_recall:.3f}')
            print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, P: {val_precision:.3f}, R: {val_recall:.3f}')
            print(f'  ICAS F1: {icas_f1:.3f}, LR: {current_lr:.6f}')
            
            # 使用ICAS F1分数保存最佳模型
            if icas_f1 > best_val_f1:
                best_val_f1 = icas_f1
                torch.save(model.state_dict(), best_model_path)
                print(f'  ✓ 新的最佳模型已保存! ICAS F1: {icas_f1:.3f}')
            
            # 早停检查
            if early_stopping(val_loss, model):
                print(f'  早停触发! 在第 {epoch+1} 轮停止训练')
                break
            
            scheduler.step()
            print()
        
        training_time = time.time() - start_time
        print(f"训练耗时: {training_time:.2f} 秒")
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_precisions': train_precisions,
            'train_recalls': train_recalls,
            'val_precisions': val_precisions,
            'val_recalls': val_recalls,
            'learning_rates': learning_rates,
            'best_val_f1': best_val_f1,
            'training_time': training_time
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return best_model_path
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs, learning_rates):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, train_accs, label='Train Acc', color='blue')
        ax2.plot(epochs, val_accs, label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, learning_rates, label='Learning Rate', color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # 训练-验证损失差异
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        ax4.plot(epochs, loss_diff, label='|Train Loss - Val Loss|', color='orange')
        ax4.set_title('Overfitting Monitor')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        print("评估模型...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = 100. * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        print(f"测试准确率: {accuracy:.2f}%")
        
        # 分类报告
        class_names = ['Non-ICAS', 'ICAS']
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, digits=4)
        print("\n分类报告:")
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存结果
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_training(self, backbone='resnet18', batch_size=32, num_epochs=50, lr=0.001,
                    weight_decay=1e-4, patience=10, use_focal_loss=True):
        """运行完整的训练流程 - 添加Focal Loss选项"""
        print("=== 热力图ICAS分类模型训练 ===\n")

        # 创建描述性输出目录
        self.output_dir = self.create_descriptive_output_dir(backbone, batch_size, num_epochs, lr, use_focal_loss)
        print(f"输出目录: {self.output_dir}")

        # 保存训练配置
        config = {
            'backbone': backbone,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'patience': patience,
            'use_focal_loss': use_focal_loss,
            'device': str(self.device),
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir)
        }

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # 1. 加载数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # 2. 计算类别权重
        class_weights = self.create_balanced_weights(y_train)
        
        # 3. 创建数据加载器
        train_dataset = ThermalDataset(X_train, y_train, self.train_transform)
        val_dataset = ThermalDataset(X_val, y_val, self.val_transform)
        test_dataset = ThermalDataset(X_test, y_test, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 4. 创建模型
        model = self.create_model(backbone=backbone)
        
        # 5. 训练模型（使用Focal Loss）
        best_model_path = self.train_model(model, train_loader, val_loader, 
                                         num_epochs, lr, weight_decay, patience, 
                                         class_weights, use_focal_loss)
        
        # 6. 评估模型
        model.load_state_dict(torch.load(best_model_path))
        results = self.evaluate_model(model, test_loader)
        
        print(f"\n训练完成! 结果保存在: {self.output_dir}")
        return results

def main():
    # 优化后的训练配置
    config = {
        'batch_size': 32,
        'num_epochs': 100,        # 增加最大轮数，让早停决定何时停止
        'learning_rate': 0.00001,
        'weight_decay': 1e-4,     # L2正则化
        'patience': 15,           # 早停耐心值
        'use_focal_loss': True,   # 使用Focal Loss
        'data_dir': './dataset/datasets/thermal_classification_cropped',
        'output_dir': './results/training_results/thermal_classifier_results'
    }
    
    # 创建训练器
    classifier = ThermalClassifier(
        data_dir=config['data_dir'],
        output_dir=config['output_dir']
    )
    
    # 开始训练 - 可以选择不同的骨干网络
    results = classifier.run_training(
        backbone='yolo11n',          # 🎯 选择骨干网络: 'resnet18', 'yolo11s', 'efficientnet_b0' 等
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        use_focal_loss=config['use_focal_loss']
    )
    
    print(f"\n最终测试准确率: {results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
