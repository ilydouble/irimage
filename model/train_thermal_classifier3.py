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
from datetime import datetime  # 添加这个导入
import matplotlib.pyplot as plt  # 添加这个导入

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
        """获取分类任务的样本 - 修复版本"""
        img_path = self.image_paths[index]
        label = self.labels[index]
    
        img = self._process_image(img_path)
        if img is None:
            # 返回零张量作为fallback
            if self.use_asymmetry_analysis:
                img = torch.zeros(6, 224, 112)  # 6通道
            else:
                img = torch.zeros(3, 224, 224)  # 3通道
    
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
            else:
                # 直接使用完整人脸进行分类
                return self.transform(face_img)
        except Exception as e:
            print(f"图像处理错误 {img_path}: {e}")
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

class RegularizedContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.1, reg_weight=0.01, reg_type='l2'):
        super(RegularizedContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.reg_weight = reg_weight
        self.reg_type = reg_type

    def forward(self, output1, output2, label, model=None):
        # 基础对比损失
        cosine_sim = F.cosine_similarity(output1, output2)
        
        contrastive_loss = torch.mean(
            label * (1 - cosine_sim) +  # 正样本：最大化相似度
            (1 - label) * torch.clamp(cosine_sim + self.margin, min=0.0)  # 负样本：最小化相似度
        )
        
        # 添加正则化项
        reg_loss = 0.0
        
        if self.reg_weight > 0 and model is not None:
            # 1. 特征范数正则化 - 防止特征过大
            feature_norm_reg = (torch.norm(output1, p=2, dim=1).mean() + 
                               torch.norm(output2, p=2, dim=1).mean()) / 2
            
            # 2. 特征分散性正则化 - 鼓励特征多样性
            batch_size = output1.size(0)
            if batch_size > 1:
                # 计算批次内特征的相似度矩阵
                sim_matrix1 = torch.matmul(output1, output1.T)
                sim_matrix2 = torch.matmul(output2, output2.T)
                
                # 移除对角线（自相似度）
                mask = ~torch.eye(batch_size, dtype=bool, device=output1.device)
                off_diag_sim1 = sim_matrix1[mask]
                off_diag_sim2 = sim_matrix2[mask]
                
                # 惩罚过高的批次内相似度
                diversity_reg = (torch.mean(torch.relu(off_diag_sim1 - 0.5)) + 
                               torch.mean(torch.relu(off_diag_sim2 - 0.5))) / 2
            else:
                diversity_reg = 0.0
            
            # 3. 权重正则化
            weight_reg = 0.0
            if self.reg_type == 'l2':
                for param in model.projection_head.parameters():
                    weight_reg += torch.norm(param, p=2)
            elif self.reg_type == 'l1':
                for param in model.projection_head.parameters():
                    weight_reg += torch.norm(param, p=1)
            
            # 组合正则化项
            reg_loss = (0.1 * feature_norm_reg + 
                       0.1 * diversity_reg + 
                       0.01 * weight_reg)
        
        total_loss = contrastive_loss + self.reg_weight * reg_loss
        
        return total_loss, contrastive_loss, reg_loss

class AdaptiveContrastiveLoss(nn.Module):
    """自适应对比损失 - 根据训练进度调整正则化强度"""
    def __init__(self, margin=0.2, temperature=0.1, max_reg_weight=0.05):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.max_reg_weight = max_reg_weight
        self.epoch = 0

    def forward(self, output1, output2, label, model=None):
        cosine_sim = F.cosine_similarity(output1, output2)
        
        contrastive_loss = torch.mean(
            label * (1 - cosine_sim) +
            (1 - label) * torch.clamp(cosine_sim + self.margin, min=0.0)
        )
        
        # 自适应正则化权重 - 训练初期强，后期弱
        current_reg_weight = self.max_reg_weight * (1.0 - min(self.epoch / 50.0, 0.8))
        
        reg_loss = 0.0
        if current_reg_weight > 0 and model is not None:
            # 特征一致性正则化 - 防止特征坍塌
            batch_size = output1.size(0)
            if batch_size > 1:
                # 计算特征的标准差
                std1 = torch.std(output1, dim=0).mean()
                std2 = torch.std(output2, dim=0).mean()
                
                # 鼓励特征有足够的方差
                variance_reg = torch.relu(0.1 - std1) + torch.relu(0.1 - std2)
                
                # 特征正交性 - 鼓励不同维度独立
                corr_matrix1 = torch.corrcoef(output1.T)
                corr_matrix2 = torch.corrcoef(output2.T)
                
                # 惩罚非对角线元素过大
                mask = ~torch.eye(output1.size(1), dtype=bool, device=output1.device)
                orthogonal_reg = (torch.mean(torch.abs(corr_matrix1[mask])) + 
                                torch.mean(torch.abs(corr_matrix2[mask]))) / 2
                
                reg_loss = variance_reg + 0.1 * orthogonal_reg
        
        total_loss = contrastive_loss + current_reg_weight * reg_loss
        
        return total_loss, contrastive_loss, reg_loss
# ======================================================================================
# 模块5: 训练器
# ======================================================================================

class ContrastiveThermalClassifier:
    def __init__(self, data_dir: str, output_dir: str = "./model/contrastive_thermal_results", 
                 use_asymmetry_analysis: bool = False, pretrained_encoder_path: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_asymmetry_analysis = use_asymmetry_analysis
        self.pretrained_encoder_path = pretrained_encoder_path  # 新增参数
        
        # 创建带时间戳的运行目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"结果将保存到: {self.run_dir}")
        print(f"不对称分析: {'启用' if use_asymmetry_analysis else '禁用'}")
        
        if pretrained_encoder_path:
            print(f"将使用预训练编码器: {pretrained_encoder_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 根据是否使用不对称分析调整尺寸
        if use_asymmetry_analysis:
            resize_size = (224, 112)  # 半脸分析
            print("使用半脸分析模式，图像尺寸: (224, 112)")
        else:
            resize_size = (224, 224)  # 完整人脸
            print("使用完整人脸模式，图像尺寸: (224, 224)")
        
        # 数据变换
        self.transform = T.Compose([
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    
    def train_contrastive_encoder(self, epochs=50, batch_size=32, lr=0.001):
        """第一阶段：对比学习训练编码器"""
        print("=== 第一阶段：对比学习训练编码器 ===")
        
        # 创建对比学习数据集
        dataset = ContrastiveThermalDataset(
            self.data_dir, 
            self.transform, 
            mode='contrastive',
            use_asymmetry_analysis=self.use_asymmetry_analysis  # 添加这个参数
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # 创建模型
        model = ThermalEncoder(backbone='resnet18').to(self.device)
        # 根据不对称分析调整模型结构
        if self.use_asymmetry_analysis:
            # 调整第一层卷积为6通道输入
            original_conv1 = model.backbone.conv1
            model.backbone.conv1 = nn.Conv2d(
                6, original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            ).to(self.device)
            print("对比学习模型已调整为6通道输入（不对称分析模式）")
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = ContrastiveLoss(margin=0.2, temperature=0.1) #调参数0.2->0.5
        
        # 早停机制 - 监控相似度差距
        best_similarity_gap = 0
        patience = 15 #调参数
        patience_counter = 0
        min_improvement = 0.001
        
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
                # 原来的监控信息
                with torch.no_grad():
                    feat1_norm = torch.norm(feat1, p=2, dim=1).mean()
                    feat2_norm = torch.norm(feat2, p=2, dim=1).mean()
                    similarities = F.cosine_similarity(feat1, feat2)
                    sim_min, sim_max = similarities.min(), similarities.max()
                    
                    if batch_idx % 50 == 0:  # 每50个batch打印一次
                        print(f'  Batch {batch_idx}: Sim range: [{sim_min:.3f}, {sim_max:.3f}]')
                    
                    # 收集正负样本相似度用于早停
                    pos_mask = labels == 1
                    neg_mask = labels == 0
                    
                    if pos_mask.sum() > 0:
                        epoch_pos_sim.extend(similarities[pos_mask].cpu().numpy())
                    if neg_mask.sum() > 0:
                        epoch_neg_sim.extend(similarities[neg_mask].cpu().numpy())
                
                # 计算损失
                loss = criterion(feat1, feat2, labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            # 计算本轮指标
            avg_loss = total_loss / len(dataloader)
            avg_pos_sim = np.mean(epoch_pos_sim) if epoch_pos_sim else 0
            avg_neg_sim = np.mean(epoch_neg_sim) if epoch_neg_sim else 0
            similarity_gap = avg_pos_sim - avg_neg_sim
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            train_losses.append(avg_loss)
            positive_similarities.append(avg_pos_sim)
            negative_similarities.append(avg_neg_sim)
            learning_rates.append(current_lr)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Loss: {avg_loss:.4f}')
            print(f'  Pos Sim: {avg_pos_sim:.4f}, Neg Sim: {avg_neg_sim:.4f}')
            print(f'  Similarity Gap: {similarity_gap:.4f}')
            
            # 早停检查
            if similarity_gap > best_similarity_gap + min_improvement:
                best_similarity_gap = similarity_gap
                patience_counter = 0
                print(f'  ✓ 新的最佳相似度差距: {similarity_gap:.4f}')
                # 保存最佳模型
                best_encoder_path = self.run_dir / "best_contrastive_encoder.pth"
                torch.save(model.state_dict(), best_encoder_path)
            else:
                patience_counter += 1
                print(f'  早停计数: {patience_counter}/{patience}')
            
            # 触发早停
            if patience_counter >= patience:
                print(f'  早停触发! 在第 {epoch+1} 轮停止训练')
                print(f'  最佳相似度差距: {best_similarity_gap:.4f}')
                break
        
        # 加载最佳模型
        best_encoder_path = self.run_dir / "best_contrastive_encoder.pth"
        if best_encoder_path.exists():
            model.load_state_dict(torch.load(best_encoder_path))
            print("已加载最佳对比学习模型")
        
        # 绘图和保存
        self.plot_contrastive_training_curves(
            train_losses, positive_similarities, negative_similarities, learning_rates
        )
        
        # 保存最终编码器
        encoder_path = self.run_dir / "contrastive_encoder.pth"
        torch.save(model.state_dict(), encoder_path)
        print(f"编码器已保存到: {encoder_path}")
        
        return model
    
    def train_classifier(self, pretrained_encoder=None, epochs=30, batch_size=32, lr=0.0001):
        """第二阶段：微调分类器 - 处理数据不平衡"""
        print("=== 第二阶段：微调分类器 ===")
        
        # 准备分类数据集
        full_dataset = ContrastiveThermalDataset(self.data_dir, self.transform, mode='classification' )
        
        # 检查数据分布
        all_labels = [full_dataset[i][1] for i in range(len(full_dataset))]
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\n=== 数据分布检查 ===")
        print(f"Non-ICAS (0): {counts[0]} 张 ({counts[0]/len(all_labels)*100:.1f}%)")
        print(f"ICAS (1): {counts[1]} 张 ({counts[1]/len(all_labels)*100:.1f}%)")
        print(f"不平衡比例: 1:{counts[0]/counts[1]:.2f}")
        
        # 数据分割 - 保持分层
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        # 使用分层采样
        from sklearn.model_selection import train_test_split
        indices = list(range(len(full_dataset)))
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, stratify=all_labels, random_state=42
        )
        temp_labels = [all_labels[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # 检查分割后的分布
        train_labels = [all_labels[i] for i in train_indices]
        val_labels = [all_labels[i] for i in val_indices]
        test_labels = [all_labels[i] for i in test_indices]
        
        print(f"\n=== 分割后数据分布 ===")
        print(f"训练集: ICAS={sum(train_labels)}, Non-ICAS={len(train_labels)-sum(train_labels)}")
        print(f"验证集: ICAS={sum(val_labels)}, Non-ICAS={len(val_labels)-sum(val_labels)}")
        print(f"测试集: ICAS={sum(test_labels)}, Non-ICAS={len(test_labels)-sum(test_labels)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        if pretrained_encoder is not None:
            model = pretrained_encoder
        else:
            model = ThermalEncoder(backbone='resnet18').to(self.device)
            
            # 如果使用不对称分析，需要调整模型结构
            if self.use_asymmetry_analysis:
                # 调整第一层卷积为6通道输入
                original_conv1 = model.backbone.conv1
                model.backbone.conv1 = nn.Conv2d(
                    6, original_conv1.out_channels, 
                    kernel_size=original_conv1.kernel_size,
                    stride=original_conv1.stride,
                    padding=original_conv1.padding,
                    bias=original_conv1.bias is not None
                ).to(self.device)
                
                # 初始化新的卷积层权重
                with torch.no_grad():
                    new_weight = torch.zeros(original_conv1.out_channels, 6, 
                                        original_conv1.kernel_size[0], 
                                        original_conv1.kernel_size[1])
                    new_weight[:, :3, :, :] = original_conv1.weight.data
                    new_weight[:, 3:, :, :] = original_conv1.weight.data
                    model.backbone.conv1.weight.data = new_weight
                
                print("分类模型已调整为6通道输入（不对称分析模式）")
            
            # 加载预训练编码器权重
            encoder_path = self.run_dir / "contrastive_encoder.pth"
            if encoder_path.exists():
                model.load_state_dict(torch.load(encoder_path, map_location=self.device))
                print("已加载预训练编码器")
        
        # 冻结编码器
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.projection_head.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # 计算类别权重处理不平衡
        from sklearn.utils.class_weight import compute_class_weight

        # 调试信息
        print(f"train_labels类型: {type(train_labels)}")
        print(f"train_labels样本: {train_labels[:10]}")
        print(f"unique train_labels: {np.unique(train_labels)}")

        # 确保数据类型一致
        train_labels_array = np.array(train_labels)
        unique_classes = np.unique(train_labels_array)

        print(f"unique_classes: {unique_classes}")
        print(f"unique_classes类型: {type(unique_classes)}")

        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels_array)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        print(f"\n=== 类别权重 ===")
        for i, weight in enumerate(class_weights):
            class_name = 'Non-ICAS' if unique_classes[i] == 0 else 'ICAS'
            print(f"{class_name} ({unique_classes[i]})权重: {weight:.3f}")
        
        # 使用加权损失函数
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # 优化器和调度器
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr*0.1, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # 早停机制 - 使用F1分数而不是准确率
        best_val_f1 = 0
        patience = 10
        patience_counter = 0
        best_model_path = self.run_dir / "best_classifier.pth"
        
        # 记录训练历史
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_preds, train_labels_epoch = [], []
            
            for img, labels in train_loader:
                img, labels = img.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model.classify(img)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels_epoch.extend(labels.cpu().numpy())
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_preds, val_labels_epoch = [], []
            
            with torch.no_grad():
                for img, labels in val_loader:
                    img, labels = img.to(self.device), labels.to(self.device)
                    outputs = model.classify(img)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels_epoch.extend(labels.cpu().numpy())
            
            # 计算指标
            train_acc = accuracy_score(train_labels_epoch, train_preds)
            val_acc = accuracy_score(val_labels_epoch, val_preds)
            train_f1 = f1_score(train_labels_epoch, train_preds, average='weighted')
            val_f1 = f1_score(val_labels_epoch, val_preds, average='weighted')
            
            # 计算ICAS类别的F1分数（更重要）
            icas_f1 = f1_score(val_labels_epoch, val_preds, pos_label=1, zero_division=0)
            
            # 记录历史
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Train F1: {train_f1s[-1]:.4f}')
            print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}, Val F1: {val_f1s[-1]:.4f}')
            print(f'  ICAS F1: {icas_f1:.4f}')
            
            # 学习率调度
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_f1)  # 使用F1分数调度
            new_lr = optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                print(f'  学习率调整: {old_lr:.6f} -> {new_lr:.6f}')
            
            # 早停检查 - 使用F1分数
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f'  ✓ 新的最佳验证F1分数: {val_f1:.4f}')
            else:
                patience_counter += 1
                print(f'  早停计数: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print(f'  早停触发! 最佳验证F1分数: {best_val_f1:.4f}')
                break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        
        # 绘图
        self.plot_classification_training_curves(
            train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s
        )
        
        # 测试最佳模型
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
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # 绘制ROC曲线（如果是二分类）
        if len(np.unique(all_labels)) == 2:
            self.plot_roc_curve(all_labels, [prob[1] for prob in all_probs])
        
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

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 类别名称
        class_names = ['Non-ICAS', 'ICAS']
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Thermal Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # 添加准确率信息
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.4f}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印分类报告
        print("\n=== 分类报告 ===")
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)

    def plot_roc_curve(self, y_true, y_scores):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, roc_auc_score
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Thermal Classification', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_full_training(self, skip_contrastive=False):
        """运行完整的两阶段训练流程"""
        print("=== 开始完整训练流程 ===")
        
        if skip_contrastive and self.pretrained_encoder_path:
            # 跳过对比学习，直接加载预训练编码器
            print("=== 跳过对比学习阶段，加载预训练编码器 ===")
            encoder = ThermalEncoder(backbone='resnet18').to(self.device)
            
            # 根据use_asymmetry_analysis调整模型结构
            if self.use_asymmetry_analysis:
                # 修改第一层卷积层以接受6通道输入
                if hasattr(encoder.backbone, 'conv1'):
                    original_conv1 = encoder.backbone.conv1
                    encoder.backbone.conv1 = nn.Conv2d(
                        6, original_conv1.out_channels, 
                        kernel_size=original_conv1.kernel_size,
                        stride=original_conv1.stride,
                        padding=original_conv1.padding,
                        bias=original_conv1.bias
                    ).to(self.device)
                print("调整模型为6通道输入（不对称分析模式）")
            
            if Path(self.pretrained_encoder_path).exists():
                encoder.load_state_dict(torch.load(self.pretrained_encoder_path, map_location=self.device))
                print(f"已加载预训练编码器: {self.pretrained_encoder_path}")
            else:
                raise FileNotFoundError(f"预训练编码器文件不存在: {self.pretrained_encoder_path}")
        else:
            # 第一阶段：对比学习（只使用训练+验证集）
            print("=== 执行完整的两阶段训练 ===")
            encoder = self.train_contrastive_encoder(epochs=100, batch_size=32, lr=0.00005)
        
        # 第二阶段：分类微调（使用训练+验证集训练，测试集评估）
        print("=== 第二阶段：分类微调 ===")
        model, test_results = self.train_classifier(encoder, epochs=30, batch_size=32, lr=0.00001)
        
        
        return model, test_results

    
    def plot_contrastive_training_curves(self, train_losses, positive_similarities, negative_similarities, learning_rates):
        """绘制对比学习训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 1. 损失曲线
        ax1.plot(epochs, train_losses, label='Contrastive Loss', color='blue', linewidth=2)
        ax1.set_title('Contrastive Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 相似度曲线
        ax2.plot(epochs, positive_similarities, label='Positive Pairs', color='green', linewidth=2)
        ax2.plot(epochs, negative_similarities, label='Negative Pairs', color='red', linewidth=2)
        ax2.set_title('Cosine Similarity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Similarity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 相似度差距
        similarity_gap = [pos - neg for pos, neg in zip(positive_similarities, negative_similarities)]
        ax3.plot(epochs, similarity_gap, label='Similarity Gap', color='purple', linewidth=2)
        ax3.set_title('Positive-Negative Similarity Gap', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gap')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习率
        ax4.plot(epochs, learning_rates, label='Learning Rate', color='orange', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'contrastive_training_curves.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.run_dir / 'classification_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# ======================================================================================
# 主程序
# ======================================================================================

def main():
    # 配置参数
    data_dir = "./dataset/datasets/thermal_classification_cropped"
    output_dir = "./model/contrastive_thermal_classifier_results"
    
    # 选项1: 完整训练（对比学习 + 分类）
    # classifier = ContrastiveThermalClassifier(data_dir, output_dir, use_asymmetry_analysis=True)
    # model, results = classifier.run_full_training(skip_contrastive=False)
    
    # 选项2: 只进行分类微调（需要指定预训练编码器路径）
    pretrained_path = "./model/contrastive_thermal_classifier_results/run_20250826_234950__/best_contrastive_encoder.pth"
    classifier = ContrastiveThermalClassifier(data_dir, output_dir, pretrained_encoder_path=pretrained_path, use_asymmetry_analysis=True)
    model, results = classifier.run_full_training(skip_contrastive=True)
    
    print(f"\n=== 最终结果 ===")
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"测试F1分数: {results['f1']:.4f}")
    print(f"测试AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()
