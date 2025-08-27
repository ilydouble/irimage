import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FocalLossClassifier:
    """实现Focal Loss的分类器包装器"""
    def __init__(self, base_classifier, alpha=0.25, gamma=2.0):
        self.base_classifier = base_classifier
        self.alpha = alpha
        self.gamma = gamma
        self.is_fitted = False
    
    def fit(self, X, y):
        """训练分类器"""
        # 对于支持sample_weight的分类器，计算focal weight
        if hasattr(self.base_classifier, 'fit') and 'sample_weight' in self.base_classifier.fit.__code__.co_varnames:
            # 先用标准方法训练一次获得初始概率
            temp_classifier = type(self.base_classifier)(**self.base_classifier.get_params())
            temp_classifier.fit(X, y)
            
            # 计算focal weights
            if hasattr(temp_classifier, 'predict_proba'):
                proba = temp_classifier.predict_proba(X)
                focal_weights = self._compute_focal_weights(y, proba)
                self.base_classifier.fit(X, y, sample_weight=focal_weights)
            else:
                self.base_classifier.fit(X, y)
        else:
            # 对于不支持sample_weight的分类器，使用类别权重
            if hasattr(self.base_classifier, 'set_params'):
                if 'class_weight' in self.base_classifier.get_params():
                    self.base_classifier.set_params(class_weight='balanced')
            self.base_classifier.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def _compute_focal_weights(self, y_true, y_proba):
        """计算focal loss权重"""
        weights = np.ones(len(y_true))
        
        for i, (true_label, proba) in enumerate(zip(y_true, y_proba)):
            # 获取真实类别的概率
            p_t = proba[true_label]
            
            # 计算alpha权重
            alpha_t = self.alpha if true_label == 1 else (1 - self.alpha)
            
            # 计算focal weight
            focal_weight = alpha_t * (1 - p_t) ** self.gamma
            weights[i] = focal_weight
        
        return weights
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("分类器尚未训练")
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("分类器尚未训练")
        if hasattr(self.base_classifier, 'predict_proba'):
            return self.base_classifier.predict_proba(X)
        else:
            # 对于不支持predict_proba的分类器，返回硬预测
            pred = self.predict(X)
            proba = np.zeros((len(pred), 2))
            proba[np.arange(len(pred)), pred] = 1.0
            return proba
    
    def get_params(self, deep=True):
        params = self.base_classifier.get_params(deep)
        params.update({'alpha': self.alpha, 'gamma': self.gamma})
        return params

class ThermalFeatureExtractor:
    def __init__(self, data_dir="./dataset/datasets/thermal_classification_cropped", 
                 output_dir="./model/thermal_feature_classifier_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别映射
        self.class_to_idx = {'non_icas': 0, 'icas': 1}
        self.idx_to_class = {0: 'non_icas', 1: 'icas'}
        
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.run_dir}")
    
    def extract_temperature_features(self, image):
        """提取温度特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        features = {}
        
        # 如果是RGB图像，需要特殊处理
        if len(image.shape) == 3:
            # 使用HSV色彩空间的H通道（色调）来表示温度
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]  # 色调通道
            saturation = hsv[:, :, 1]  # 饱和度通道
            value = hsv[:, :, 2]  # 明度通道
            
            # 创建掩码排除黑色背景
            mask = (saturation > 30) & (value > 30)  # 排除低饱和度和低明度的背景
            
            if np.sum(mask) == 0:
                print("警告: 图像全为背景")
                return {}
            
            # 使用色调作为温度代理
            temp_proxy = hue[mask]
            
        else:
            # 如果是单通道图像
            mask = gray > 1  # 排除背景像素
            if np.sum(mask) == 0:
                return {}
            temp_proxy = gray[mask]
        
        # 基本统计特征 - 只计算前景区域
        features['temp_mean'] = np.mean(temp_proxy)
        features['temp_std'] = np.std(temp_proxy)
        features['temp_min'] = np.min(temp_proxy)
        features['temp_max'] = np.max(temp_proxy)
        features['temp_median'] = np.median(temp_proxy)
        features['temp_range'] = features['temp_max'] - features['temp_min']
        features['temp_var'] = np.var(temp_proxy)
        
        # 百分位数特征
        features['temp_p25'] = np.percentile(temp_proxy, 25)
        features['temp_p75'] = np.percentile(temp_proxy, 75)
        features['temp_iqr'] = features['temp_p75'] - features['temp_p25']
        
        # 偏度和峰度
        from scipy.stats import skew, kurtosis
        features['temp_skewness'] = skew(temp_proxy)
        features['temp_kurtosis'] = kurtosis(temp_proxy)
        
        # 熵 - 只计算前景区域
        features['temp_entropy'] = shannon_entropy(temp_proxy.astype(np.uint8))
        
        # 前景区域比例
        features['foreground_ratio'] = np.sum(mask) / mask.size
        
        # 梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        features['gradient_max'] = np.max(gradient_magnitude)
        
        # 温度分布特征
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist / np.sum(hist)  # 归一化
        for i, val in enumerate(hist):
            features[f'temp_hist_{i}'] = val
        
        return features
    
    def extract_block_lbp_features(self, image, n_blocks=(4, 4)):
        """提取分块LBP特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        features = {}
        
        # 参数设置
        radius = 2
        n_points = 8 * radius
        method = 'uniform'
        
        try:
            # 计算全局LBP
            lbp_image = local_binary_pattern(gray, n_points, radius, method)
            
            # 获取图像尺寸
            h, w = gray.shape
            block_h, block_w = h // n_blocks[0], w // n_blocks[1]
            
            # 分块计算直方图
            for i in range(n_blocks[0]):
                for j in range(n_blocks[1]):
                    # 提取当前块
                    block = lbp_image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    
                    # 计算当前块的直方图 (18 bins for uniform LBP)
                    hist, _ = np.histogram(block.ravel(), bins=np.arange(0, n_points+3), 
                                         range=(0, n_points+2), density=True)
                    
                    # 为每个块的每个bin创建特征名
                    for bin_idx, hist_val in enumerate(hist):
                        features[f'block_lbp_{i}_{j}_bin_{bin_idx}'] = hist_val
            
            # 添加全局统计量
            features['global_lbp_mean'] = np.mean(lbp_image)
            features['global_lbp_std'] = np.std(lbp_image)
            features['global_lbp_entropy'] = shannon_entropy(lbp_image.astype(np.uint8))
            
            # 添加块间统计特征
            block_means = []
            for i in range(n_blocks[0]):
                for j in range(n_blocks[1]):
                    block = lbp_image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    block_means.append(np.mean(block))
            
            features['block_lbp_mean_var'] = np.var(block_means)  # 块间均值方差
            features['block_lbp_mean_range'] = np.max(block_means) - np.min(block_means)  # 块间均值范围
            
        except Exception as e:
            print(f"分块LBP计算错误: {e}")
            # 返回默认值
            total_bins = n_blocks[0] * n_blocks[1] * (n_points + 2)
            for i in range(total_bins):
                features[f'block_lbp_feature_{i}'] = 0.0
            features['global_lbp_mean'] = 0.0
            features['global_lbp_std'] = 0.0
            features['global_lbp_entropy'] = 0.0
            features['block_lbp_mean_var'] = 0.0
            features['block_lbp_mean_range'] = 0.0
        
        return features
    
    def extract_texture_features(self, image):
        """提取纹理特征 - 使用分块LBP替代原有纹理特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        features = {}
        
        # 使用分块LBP特征替代原有的GLCM和LBP特征
        block_lbp_features = self.extract_block_lbp_features(image, n_blocks=(4, 4))
        features.update(block_lbp_features)
        
        # 可选：保留一些简单的纹理特征作为补充
        # 梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        features['gradient_max'] = np.max(gradient_magnitude)
        
        return features
    
    def extract_shape_features(self, image):
        """提取形状特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        features = {}
        
        # 多阈值二值化
        thresholds = [cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
                     cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE]
        
        for i, thresh_type in enumerate(thresholds):
            try:
                _, binary = cv2.threshold(gray, 0, 255, thresh_type)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    
                    features[f'contour_area_{i}'] = area
                    features[f'contour_perimeter_{i}'] = perimeter
                    features[f'contour_compactness_{i}'] = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
                    
                    # 边界框
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    features[f'bbox_aspect_ratio_{i}'] = w / h if h > 0 else 0
                    features[f'bbox_extent_{i}'] = area / (w * h) if (w * h) > 0 else 0
                else:
                    features[f'contour_area_{i}'] = 0
                    features[f'contour_perimeter_{i}'] = 0
                    features[f'contour_compactness_{i}'] = 0
                    features[f'bbox_aspect_ratio_{i}'] = 1
                    features[f'bbox_extent_{i}'] = 0
            except:
                features[f'contour_area_{i}'] = 0
                features[f'contour_perimeter_{i}'] = 0
                features[f'contour_compactness_{i}'] = 0
                features[f'bbox_aspect_ratio_{i}'] = 1
                features[f'bbox_extent_{i}'] = 0
        
        return features
    
    def extract_all_features(self, image_path):
        """提取所有特征"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        features = {}
        
        # 提取各类特征
        temp_features = self.extract_temperature_features(image)
        texture_features = self.extract_texture_features(image)
        shape_features = self.extract_shape_features(image)
        
        # 合并所有特征
        features.update(temp_features)
        features.update(texture_features)
        features.update(shape_features)
        
        # 添加图像基本信息
        features['image_height'], features['image_width'] = image.shape[:2]
        features['image_channels'] = image.shape[2] if len(image.shape) == 3 else 1
        features['image_area'] = features['image_height'] * features['image_width']
        
        return features
    
    def load_and_extract_features(self):
        """加载数据并提取特征"""
        print("开始提取特征...")
        
        all_features = []
        all_labels = []
        all_filenames = []
        
        # 遍历每个类别
        for class_name in ['icas', 'non_icas']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: 类别目录不存在 {class_dir}")
                continue
            
            class_label = self.class_to_idx[class_name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            print(f"处理类别 {class_name}: {len(image_files)} 张图像")
            
            for img_path in tqdm(image_files, desc=f"提取 {class_name} 特征"):
                features = self.extract_all_features(img_path)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(class_label)
                    all_filenames.append(img_path.name)
        
        # 转换为DataFrame
        features_df = pd.DataFrame(all_features)
        features_df['label'] = all_labels
        features_df['filename'] = all_filenames
        
        print(f"特征提取完成!")
        print(f"总样本数: {len(features_df)}")
        print(f"特征维度: {len(features_df.columns) - 2}")
        print(f"ICAS: {sum(all_labels)} 张")
        print(f"Non-ICAS: {len(all_labels) - sum(all_labels)} 张")
        print(f"类别不平衡比例: 1:{(len(all_labels) - sum(all_labels))/max(sum(all_labels), 1):.2f}")
        
        # 保存特征
        features_path = self.run_dir / "extracted_features.csv"
        features_df.to_csv(features_path, index=False)
        print(f"特征已保存到: {features_path}")
        
        return features_df
    
    def prepare_data(self, features_df):
        """准备训练数据"""
        # 分离特征和标签
        feature_columns = [col for col in features_df.columns if col not in ['label', 'filename']]
        X = features_df[feature_columns]
        y = features_df['label']
        
        # 处理缺失值和无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # 移除方差为0的特征
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selector.variances_[i] > 0.01]
        
        print(f"特征选择: {len(feature_columns)} -> {len(selected_features)}")
        
        # 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_selected, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"数据集划分:")
        print(f"训练集: {len(X_train)} (ICAS: {sum(y_train)}, Non-ICAS: {len(y_train) - sum(y_train)})")
        print(f"验证集: {len(X_val)} (ICAS: {sum(y_val)}, Non-ICAS: {len(y_val) - sum(y_val)})")
        print(f"测试集: {len(X_test)} (ICAS: {sum(y_test)}, Non-ICAS: {len(y_test) - sum(y_test)})")
        
        # 保存预处理器
        scaler_path = self.run_dir / "feature_scaler.pkl"
        selector_path = self.run_dir / "feature_selector.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(selector_path, 'wb') as f:
            pickle.dump(selector, f)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, selected_features, scaler, selector)
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """训练多种机器学习模型"""
        print("开始训练多种机器学习模型...")
        
        # 计算类别权重
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        print(f"类别权重: {class_weight_dict}")
        
        # 定义基础模型
        base_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'XGBoost': None,  # 如果安装了xgboost
        }
        
        # 尝试添加XGBoost
        try:
            import xgboost as xgb
            scale_pos_weight = class_weights[0] / class_weights[1]
            base_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
        except ImportError:
            print("XGBoost未安装，跳过XGBoost模型")
            del base_models['XGBoost']
        
        # 创建Focal Loss包装的模型
        models = {}
        for name, base_model in base_models.items():
            if base_model is not None:
                models[name] = base_model
                models[f'{name}_Focal'] = FocalLossClassifier(
                    base_classifier=type(base_model)(**base_model.get_params()),
                    alpha=0.25,
                    gamma=2.0
                )
        
        # 训练和评估模型
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            start_time = time.time()
            
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 验证集预测
                val_pred = model.predict(X_val)
                val_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 计算指标
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_val, val_pred)
                precision = precision_score(y_val, val_pred, zero_division=0)
                recall = recall_score(y_val, val_pred, zero_division=0)
                f1 = f1_score(y_val, val_pred, zero_division=0)
                
                auc = roc_auc_score(y_val, val_pred_proba) if val_pred_proba is not None else 0
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'training_time': training_time
                }
                
                trained_models[name] = model
                
                print(f"  准确率: {accuracy:.4f}")
                print(f"  精确率: {precision:.4f}")
                print(f"  召回率: {recall:.4f}")
                print(f"  F1分数: {f1:.4f}")
                print(f"  AUC: {auc:.4f}")
                print(f"  训练时间: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"  训练失败: {e}")
                results[name] = None
        
        return results, trained_models
    
    def evaluate_best_model(self, trained_models, results, X_test, y_test):
        """评估最佳模型"""
        # 导入所需的评估函数
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 选择F1分数最高的模型
        valid_results = {k: v for k, v in results.items() if v is not None}
        best_model_name = max(valid_results.keys(), 
                            key=lambda x: valid_results[x]['f1_score'])
        best_model = trained_models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"验证集F1分数: {valid_results[best_model_name]['f1_score']:.4f}")
        
        # 测试集评估
        test_pred = best_model.predict(X_test)
        test_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # 详细评估报告
        print("\n=== 测试集评估结果 ===")
        report = classification_report(y_test, test_pred, target_names=['Non-ICAS', 'ICAS'], output_dict=True)
        print(classification_report(y_test, test_pred, target_names=['Non-ICAS', 'ICAS']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-ICAS', 'ICAS'], 
                   yticklabels=['Non-ICAS', 'ICAS'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.run_dir / f'confusion_matrix_{best_model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC曲线
        if test_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, test_pred_proba)
            auc_score = roc_auc_score(y_test, test_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.run_dir / f'roc_curve_{best_model_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 保存最佳模型
        model_path = self.run_dir / f'best_model_{best_model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"最佳模型已保存到: {model_path}")
        
        # 返回测试结果
        test_results = {
            'best_model': best_model_name,
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred, zero_division=0),
            'test_recall': recall_score(y_test, test_pred, zero_division=0),
            'test_f1': f1_score(y_test, test_pred, zero_division=0),
            'test_auc': auc_score if test_pred_proba is not None else 0,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return best_model_name, best_model, test_results
    
    def save_results(self, results, test_results, feature_columns, config):
        """保存训练结果"""
        # 保存模型比较结果
        results_df = pd.DataFrame(results).T
        results_path = self.run_dir / "model_comparison.csv"
        results_df.to_csv(results_path)
        
        # 保存完整结果
        full_results = {
            'config': config,
            'data_info': {
                'data_dir': str(self.data_dir),
                'num_features': len(feature_columns),
                'feature_columns': feature_columns
            },
            'validation_results': results,
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.run_dir / 'full_results.json', 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        # 保存特征重要性（如果最佳模型支持）
        best_model_name = test_results['best_model']
        if best_model_name in results and hasattr(results[best_model_name], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': results[best_model_name].feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(self.run_dir / 'feature_importance.csv', index=False)
            
            # 可视化特征重要性
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importance - {best_model_name}')
            plt.tight_layout()
            plt.savefig(self.run_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"所有结果已保存到: {self.run_dir}")
    
    def run_training(self, config=None):
        """运行完整的训练流程"""
        if config is None:
            config = {
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'random_state': 42,
                'test_size': 0.3,
                'val_size': 0.5
            }
        
        print("=== 基于特征提取的热力图ICAS分类 (Focal Loss) ===\n")
        
        start_time = time.time()
        
        # 1. 提取特征
        features_df = self.load_and_extract_features()
        
        # 2. 准备数据
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         feature_columns, scaler, selector) = self.prepare_data(features_df)
        
        # 3. 训练模型
        results, trained_models = self.train_models(X_train, X_val, y_train, y_val)
        
        # 4. 评估最佳模型
        best_model_name, best_model, test_results = self.evaluate_best_model(
            trained_models, results, X_test, y_test)
        
        total_time = time.time() - start_time
        config['total_training_time'] = total_time
        
        # 5. 保存结果
        self.save_results(results, test_results, feature_columns, config)
        
        print(f"\n=== 训练完成 ===")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"最佳模型: {best_model_name}")
        print(f"测试集F1分数: {test_results['test_f1']:.4f}")
        print(f"结果保存在: {self.run_dir}")
        
        return results, best_model_name, test_results

def main():
    # 配置参数
    config = {
        'focal_loss_alpha': 0.25,  # Focal Loss alpha参数
        'focal_loss_gamma': 2.0,   # Focal Loss gamma参数
        'random_state': 42,
        'test_size': 0.3,
        'val_size': 0.5
    }
    
    # 创建特征提取器和分类器
    classifier = ThermalFeatureExtractor(
        data_dir="./dataset/datasets/thermal_classification_cropped",
        output_dir="./model/thermal_feature_classifier_results"
    )
    
    # 运行训练
    results, best_model, test_results = classifier.run_training(config)
    
    print(f"\n=== 最终结果 ===")
    print(f"最佳模型: {best_model}")
    print(f"测试准确率: {test_results['test_accuracy']:.4f}")
    print(f"测试F1分数: {test_results['test_f1']:.4f}")
    print(f"测试AUC: {test_results['test_auc']:.4f}")

if __name__ == "__main__":
    main()
