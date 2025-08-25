import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import os
import random
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ======================================================================================
# 模块1: 面部分割 (复用train_siamese.py)
# ======================================================================================

def preprocess_and_split_face(
    image: Image.Image, 
    output_size: Tuple[int, int] = (224, 112)
) -> Optional[Tuple[Image.Image, Image.Image]]:
    """直接对已裁剪的人脸图像进行左右分割"""
    width, height = image.size
    center_x = width // 2
    
    left_half = image.crop((0, 0, center_x, height))
    right_half = image.crop((center_x, 0, width, height))
    
    if left_half.size[0] == 0 or right_half.size[0] == 0:
        return None
        
    return left_half, right_half

# ======================================================================================
# 模块2: 对比学习数据集
# ======================================================================================

class ContrastiveThermalDataset(Dataset):
    def __init__(self, root_dir: str, transform: T.Compose, mode: str = 'train', use_asymmetry_analysis: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.use_asymmetry_analysis = use_asymmetry_analysis  # 添加这个属性
        self.class_to_images: Dict[str, List[str]] = self._find_classes()
        self.classes = list(self.class_to_images.keys())
        
        # 为分类任务准备数据
        self.image_paths = []
        self.labels = []
        for class_idx, class_name in enumerate(self.classes):
            for img_path in self.class_to_images[class_name]:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def _find_classes(self) -> Dict[str, List[str]]:
        class_to_images = {}
        for class_name in os.listdir(self.root_dir):
            class_dir = self.root_dir / class_name
            if class_dir.is_dir():
                images = [str(class_dir / img) for img in os.listdir(class_dir) 
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(images) > 1:
                    class_to_images[class_name] = images
        return class_to_images

    def __len__(self) -> int:
        if self.mode == 'contrastive':
            return len(self.classes) * 200  # 增加到200，每轮400个对比对
        else:
            return len(self.image_paths)

    def __getitem__(self, index: int):
        if self.mode == 'contrastive':
            return self._get_contrastive_pair(index)
        else:
            return self._get_classification_sample(index)
    
    def _get_contrastive_pair(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成对比学习的样本对 - 添加调试信息"""
        attempts = 0
        while attempts < 10:  # 限制尝试次数
            try:
                # 决定是正样本对还是负样本对
                if index % 2 == 0:
                    # 正样本对 (同类)
                    label = 1.0
                    class_name = random.choice(self.classes)
                    img_path1, img_path2 = random.sample(self.class_to_images[class_name], 2)
                else:
                    # 负样本对 (不同类)
                    label = 0.0
                    class1, class2 = random.sample(self.classes, 2)
                    img_path1 = random.choice(self.class_to_images[class1])
                    img_path2 = random.choice(self.class_to_images[class2])
                
                # 加载并处理图像
                img1 = self._process_image(img_path1)
                img2 = self._process_image(img_path2)
                
                # 添加数据验证
                if img1 is None or img2 is None:
                    attempts += 1
                    continue
                
                # 检查张量是否有效
                if torch.isnan(img1).any() or torch.isnan(img2).any():
                    attempts += 1
                    continue
                
                return img1, img2, torch.tensor(label, dtype=torch.float32)
            
            except Exception as e:
                print(f"数据加载错误: {e}")
                attempts += 1
                continue
        
        # 如果多次尝试失败，返回零张量
        return torch.zeros(3, 224, 112), torch.zeros(3, 224, 112), torch.tensor(0.0)
    
    def _get_classification_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取分类任务的样本"""
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        img = self._process_image(img_path)
        if img is None:
            # 返回零张量作为fallback
            img = torch.zeros(3, 224, 112)
        
        return img, torch.tensor(label, dtype=torch.long)
    
    def _process_image(self, img_path: str) -> Optional[torch.Tensor]:
        """处理单张图像 - 改进版本"""
        try:
            face_img = Image.open(img_path).convert("RGB")
            
            if self.use_asymmetry_analysis:
                # 如果需要分析不对称性
                halves = preprocess_and_split_face(face_img)
                if halves is None:
                    return None
                left_half, right_half = halves
                
                # 方案1: 拼接左右脸
                left_tensor = self.transform(left_half)
                right_tensor = self.transform(right_half)
                # 在通道维度拼接 (3+3=6通道)
                return torch.cat([left_tensor, right_tensor], dim=0)
                
                # 方案2: 只用左脸 (当前的做法)
                # return self.transform(left_half)
                
                # 方案3: 计算差异特征 (需要修改网络结构)
                # return self.compute_asymmetry_features(left_half, right_half)
            else:
                # 直接使用完整人脸进行分类
                return self.transform(face_img)
        except:
            return None

# ======================================================================================
# 模块3: 孪生网络编码器
# ======================================================================================

class ThermalEncoder(nn.Module):
    """热力图特征编码器"""
    def __init__(self, backbone: str = 'resnet18', feature_dim: int = 512):
        super(ThermalEncoder, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")
        
        # 投影头用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 二分类
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        
        if return_features:
            return features
        
        # 对比学习时返回投影特征
        projected = self.projection_head(features)
        # 确保L2归一化
        return F.normalize(projected, p=2, dim=1)
    
    def classify(self, x):
        """分类任务的前向传播"""
        features = self.backbone(x)
        return self.classifier(features)

# ======================================================================================
# 模块4: 损失函数
# ======================================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.1):  # 降低margin
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, output1, output2, label):
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(output1, output2)
        
        # 改进的对比损失
        loss_contrastive = torch.mean(
            label * (1 - cosine_sim) +  # 正样本：最大化相似度
            (1 - label) * torch.clamp(cosine_sim + self.margin, min=0.0)  # 负样本：最小化相似度
        )
        
        return loss_contrastive

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线
        mask = mask - torch.eye(batch_size).to(mask.device)
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss

class InfoNCELoss(nn.Module):
    """InfoNCE损失函数 - 更稳定的对比学习损失"""
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2, labels):
        # 计算相似度矩阵
        batch_size = features1.size(0)
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # 创建标签矩阵
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))
        
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()

# ======================================================================================
# 模块5: 训练器
# ======================================================================================

class ContrastiveThermalClassifier:
    def __init__(self, data_dir: str, output_dir: str = "./model/contrastive_thermal_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 计算真实的数据集统计值
        mean, std = self.calculate_dataset_stats(str(data_dir))
        
        # 使用真实统计值的数据变换
        self.transform = T.Compose([
            T.Resize((224, 112)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)  # 使用真实统计值
        ])
    
    def train_contrastive_encoder(self, epochs=50, batch_size=32, lr=0.001):
        """第一阶段：对比学习训练编码器"""
        print("=== 第一阶段：对比学习训练编码器 ===")
        
        # 创建对比学习数据集
        dataset = ContrastiveThermalDataset(self.data_dir, self.transform, mode='contrastive')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # 创建模型
        model = ThermalEncoder(backbone='resnet18').to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = ContrastiveLoss(margin=0.5, temperature=0.1)
        
        # 记录训练历史
        train_losses = []
        positive_similarities = []
        negative_similarities = []
        learning_rates = []
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            epoch_pos_sim = []
            epoch_neg_sim = []
            
            for batch_idx, (img1, img2, labels) in enumerate(dataloader):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 获取特征
                feat1 = model(img1)
                feat2 = model(img2)
                
                # 计算相似度用于监控
                with torch.no_grad():
                    similarities = F.cosine_similarity(feat1, feat2)
                    pos_mask = labels == 1
                    neg_mask = labels == 0
                    
                    if pos_mask.sum() > 0:
                        epoch_pos_sim.extend(similarities[pos_mask].cpu().numpy())
                    if neg_mask.sum() > 0:
                        epoch_neg_sim.extend(similarities[neg_mask].cpu().numpy())
                
                # 计算损失
                loss = criterion(feat1, feat2, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        # 检查特征范数
                        feat1_norm = torch.norm(feat1, dim=1).mean()
                        feat2_norm = torch.norm(feat2, dim=1).mean()
                        
                        # 检查相似度分布
                        similarities = F.cosine_similarity(feat1, feat2)
                        
                        print(f'Batch {batch_idx}:')
                        print(f'  Loss: {loss.item():.4f}')
                        print(f'  Feat1 norm: {feat1_norm:.4f}')
                        print(f'  Feat2 norm: {feat2_norm:.4f}')
                        print(f'  Sim range: [{similarities.min():.3f}, {similarities.max():.3f}]')
                        print(f'  Pos/Neg ratio: {labels.mean():.2f}')
                
                if batch_idx % 20 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # 记录每轮统计
            avg_loss = total_loss / len(dataloader)
            avg_pos_sim = np.mean(epoch_pos_sim) if epoch_pos_sim else 0
            avg_neg_sim = np.mean(epoch_neg_sim) if epoch_neg_sim else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            train_losses.append(avg_loss)
            positive_similarities.append(avg_pos_sim)
            negative_similarities.append(avg_neg_sim)
            learning_rates.append(current_lr)
            
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
            print(f'  Pos Similarity: {avg_pos_sim:.4f}')
            print(f'  Neg Similarity: {avg_neg_sim:.4f}')
            print(f'  Similarity Gap: {avg_pos_sim - avg_neg_sim:.4f}')
        
        # 调用绘图函数
        self.plot_contrastive_training_curves(
            train_losses, positive_similarities, negative_similarities, learning_rates
        )
        
        # 保存预训练的编码器
        encoder_path = self.output_dir / "contrastive_encoder.pth"
        torch.save(model.state_dict(), encoder_path)
        print(f"编码器已保存到: {encoder_path}")
        
        return model
    
    def train_classifier(self, pretrained_encoder=None, epochs=30, batch_size=32, lr=0.0001):
        """第二阶段：微调分类器"""
        print("=== 第二阶段：微调分类器 ===")
        
        # 准备分类数据集
        full_dataset = ContrastiveThermalDataset(self.data_dir, self.transform, mode='classification')
        
        # 数据分割
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        if pretrained_encoder is not None:
            model = pretrained_encoder
        else:
            model = ThermalEncoder(backbone='resnet18').to(self.device)
            # 加载预训练编码器
            encoder_path = self.output_dir / "contrastive_encoder.pth"
            if encoder_path.exists():
                model.load_state_dict(torch.load(encoder_path))
                print("已加载预训练编码器")
        
        # 冻结backbone，只训练分类头
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 记录训练历史
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        
        # 记录训练历史
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        
        best_val_acc = 0
        best_model_path = self.output_dir / "best_classifier.pth"
        train_preds, train_labels = [], []
            
            
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            for img, labels in train_loader:
                img, labels = img.to(self.device), labels.to(self.device)
                
                _, predicted = torch.max(outputs.data, 1)    optimizer.zero_grad()
                train_preds.extend(predicted.cpu().numpy())outputs = model.classify(img)
                train_lab lsosxtend(l be=s.cpu c.numpy())riterion(outputs, labels)
            
            # 验证
            model.e   ()rd()
                mossizer.step()
                preds, va_label[], []
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # 验证
            model.eval()
            val_loss = 0
            val_preds, val_labels = [], []
            preds.exend(prediced.cpu().numpy())
                  va_lxtendlabels.cpu(.numpy())
            
with tor    # 计算指标
            train_acc = accuracy_score(train_labels, train_preds)
            ch.naoc = accuracy_sc_ge(val_labels, val_prads)
            (rain_f1)f1_scoretrain_labels, train_s, average='wegh')
            val_f1f1_score(val_ls, va_pred, average='weighted'
            
            # 记录历史
            train_lossesappend(train_los / lentrain_loader)
            val_lossesappend(val_loss / lnval_loader)
            train_accs.append(train_acc)    for img, labels in val_loader:
                   s.append(g, laca)
            tlain_f1s.appsnd(=rain_f1)
           mg.tof1s.append(vlf_f1)
            .device), labels.to(self.device)
                    outputs = model.classify(img)s[-1]
                    loss = criterion(ou)s[-1]
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy()))
        
        # 调用绘图函数
        self.plot_classification_training_curves(
            train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s
        
            
            # 计算指标
            train_acc = accuracy_score(train_labels, train_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            # 记录历史
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
        
        # 调用绘图函数
        self.plot_classification_training_curves(
            train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s
        )
        
        # 测试最佳模型
        model.load_state_dict(torch.load(best_model_path))
        test_results = self.evaluate_model(model, test_loader)
        
        return model, test_results
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for img, labels in test_loader:
                img, labels = img.to(self.device), labels.to(self.device)
                outputs = model.classify(img)
                probs = F.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # AUC (二分类)
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
        else:
            auc = 0.0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"\n=== 测试结果 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return results
    
    def run_full_training(self):
        """运行完整的两阶段训练"""
        print("开始对比学习热力图分类训练...")
        
        # 第一阶段：对比学习
        encoder = self.train_contrastive_encoder(epochs=100, batch_size=32, lr=0.005)
        
        # 第二阶段：分类微调
        model, test_results = self.train_classifier(encoder, epochs=30, batch_size=32, lr=0.0005)
        
        return model, test_results

    def plot_contrastive_training_curves(self, train_losses, pos_similarities, neg_similarities, learning_rates):
        """绘制对比学习训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 损失曲线
        ax1.plot(epochs, train_losses, label='Contrastive Loss', color='blue', linewidth=2)
        ax1.set_title('Contrastive Learning Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 相似度曲线
        ax2.plot(epochs, pos_similarities, label='Positive Pairs', color='green', linewidth=2)
        ax2.plot(epochs, neg_similarities, label='Negative Pairs', color='red', linewidth=2)
        similarity_gap = [p - n for p, n in zip(pos_similarities, neg_similarities)]
        ax2.plot(epochs, similarity_gap, label='Similarity Gap', color='purple', linewidth=2, linestyle='--')
        ax2.set_title('Cosine Similarity Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cosine Similarity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. 学习率曲线
        ax3.plot(epochs, learning_rates, label='Learning Rate', color='orange', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. 相似度分布热力图
        if len(pos_similarities) > 10:  # 确保有足够数据
            # 创建相似度变化的热力图
            similarity_matrix = np.array([pos_similarities, neg_similarities])
            im = ax4.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
            ax4.set_title('Similarity Evolution Heatmap', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Pair Type')
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['Positive', 'Negative'])
            plt.colorbar(im, ax=ax4, label='Cosine Similarity')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'contrastive_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_classification_training_curves(self, train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
        """绘制分类训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 损失曲线
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        ax1.plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
        ax1.set_title('Classification Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2.plot(epochs, train_accs, label='Train Acc', color='blue', linewidth=2)
        ax2.plot(epochs, val_accs, label='Val Acc', color='red', linewidth=2)
        ax2.set_title('Classification Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. F1分数曲线
        ax3.plot(epochs, train_f1s, label='Train F1', color='green', linewidth=2)
        ax3.plot(epochs, val_f1s, label='Val F1', color='orange', linewidth=2)
        ax3.set_title('F1 Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 过拟合监控
        loss_gap = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        acc_gap = [abs(t - v) for t, v in zip(train_accs, val_accs)]
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, loss_gap, label='Loss Gap', color='red', linewidth=2)
        line2 = ax4_twin.plot(epochs, acc_gap, label='Acc Gap', color='blue', linewidth=2)
        
        ax4.set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference', color='red')
        ax4_twin.set_ylabel('Accuracy Difference', color='blue')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# ======================================================================================
# 主程序
# ======================================================================================

def main():
    # 配置参数
    data_dir = "./dataset/datasets/thermal_classification_cropped"
    output_dir = "./model/contrastive_thermal_classifier_results"
    
    # 创建分类器
    classifier = ContrastiveThermalClassifier(data_dir, output_dir)
    
    # 运行训练
    model, results = classifier.run_full_training()
    
    print(f"\n=== 最终结果 ===")
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"测试F1分数: {results['f1']:.4f}")
    print(f"测试AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()
