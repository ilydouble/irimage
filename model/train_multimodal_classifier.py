import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import time
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
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

class MultiModalThermalClassifier:
    def __init__(self,
                 data_dir="./dataset/datasets/thermal_classification_cropped",
                 db_path="./web/database/patientcare.db",
                 output_dir="./results/training_results/multimodal_thermal_classifier_results"):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.base_output_dir = Path(output_dir)
        
        # 类别映射
        self.class_to_idx = {'non_icas': 0, 'icas': 1}
        self.idx_to_class = {0: 'non_icas', 1: 'icas'}

        print(f"数据目录: {self.data_dir}")
        print(f"数据库路径: {self.db_path}")
        print(f"基础输出目录: {self.base_output_dir}")

    def create_descriptive_output_dir(self, best_model_name, focal_alpha, focal_gamma):
        """创建描述性的输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 格式: MULTIMODAL-RF-FOCAL-A0.25-G2.0-20240829_143022
        folder_name = f"MULTIMODAL-{best_model_name.upper()}-FOCAL-A{focal_alpha}-G{focal_gamma}-{timestamp}"

        output_dir = self.base_output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir
    
    def load_clinical_features(self):
        """从数据库加载临床特征"""
        print("加载临床特征数据...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 查询患者临床特征
            query = """
            SELECT patient_id, age, gender, height, weight, waist, hip, neck, bmi, has_icas
            FROM patients
            WHERE patient_id IS NOT NULL
            """
            
            clinical_df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"从数据库加载了 {len(clinical_df)} 个患者的临床特征")
            print(f"临床特征列: {list(clinical_df.columns)}")
            
            return clinical_df
            
        except Exception as e:
            print(f"加载临床特征失败: {e}")
            return pd.DataFrame()
    
    def encode_clinical_features(self, clinical_df):
        """编码和预处理临床特征"""
        print("编码临床特征...")
        
        # 复制数据避免修改原始数据
        df = clinical_df.copy()
        
        # 处理缺失值
        numeric_columns = ['age', 'height', 'weight', 'waist', 'hip', 'neck', 'bmi']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # 性别编码
        if 'gender' in df.columns:
            df['gender_encoded'] = df['gender'].map({'male': 0, 'female': 1}).fillna(0)
        
        # 计算BMI（如果没有的话）
        if 'bmi' not in df.columns or df['bmi'].isna().sum() > 0:
            df['bmi_calculated'] = df['weight'] / ((df['height'] / 100) ** 2)
            df['bmi'] = df['bmi'].fillna(df['bmi_calculated'])
        
        # 创建衍生特征
        df['waist_hip_ratio'] = df['waist'] / df['hip']
        df['waist_height_ratio'] = df['waist'] / df['height']
        df['neck_height_ratio'] = df['neck'] / df['height']
        
        # BMI分类特征
        def bmi_category(bmi):
            if pd.isna(bmi):
                return 0
            elif bmi < 18.5:
                return 0  # 偏瘦
            elif bmi < 24:
                return 1  # 正常
            elif bmi < 28:
                return 2  # 超重
            else:
                return 3  # 肥胖
        
        df['bmi_category'] = df['bmi'].apply(bmi_category)
        
        # 年龄分组
        def age_group(age):
            if pd.isna(age):
                return 0
            elif age < 30:
                return 0
            elif age < 45:
                return 1
            elif age < 60:
                return 2
            else:
                return 3
        
        df['age_group'] = df['age'].apply(age_group)
        
        # 选择最终的特征列 - 移除目标变量has_icas
        feature_columns = [
            'age', 'gender_encoded', 'height', 'weight', 'waist', 'hip', 'neck', 'bmi',
            'waist_hip_ratio', 'waist_height_ratio', 'neck_height_ratio',
            'bmi_category', 'age_group'
        ]
        
        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in df.columns]
        
        print(f"可用的临床特征: {available_features}")
        
        # 返回时保留has_icas作为标签，但不包含在特征中
        return df[['patient_id', 'has_icas'] + available_features], available_features
    
    def extract_image_features(self, image_path):
        """提取图像特征（复用之前的方法）"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        features = {}
        
        # 提取温度特征
        temp_features = self.extract_temperature_features(image)
        features.update(temp_features)
        
        # 提取纹理特征
        texture_features = self.extract_texture_features(image)
        features.update(texture_features)
        
        # 提取形状特征
        shape_features = self.extract_shape_features(image)
        features.update(shape_features)
        
        # 添加图像基本信息
        # features['image_height'], features['image_width'] = image.shape[:2]
        # features['image_channels'] = image.shape[2] if len(image.shape) == 3 else 1
        #features['image_area'] = features['image_height'] * features['image_width']
        
        return features
    
    def extract_temperature_features(self, image):
        """提取温度特征 - 针对RGB热力图"""
        features = {}
        
        # 如果是RGB图像，需要特殊处理
        if len(image.shape) == 3:
            # 方法1: 使用HSV色彩空间的H通道（色调）来表示温度
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]  # 色调通道
            saturation = hsv[:, :, 1]  # 饱和度通道
            value = hsv[:, :, 2]  # 明度通道
            
            # 创建掩码排除黑色背景
            # 使用饱和度和明度来识别背景
            mask = (saturation > 30) & (value > 30)  # 排除低饱和度和低明度的背景
            
            if np.sum(mask) == 0:
                print("警告: 图像全为背景")
                return {}
            
            # 使用色调作为温度代理
            temp_proxy = hue[mask]
            
            # 方法2: 也可以使用RGB通道的组合
            # 红色通道通常表示高温区域
            red_channel = image[:, :, 2]  # BGR格式中红色是第2个通道
            blue_channel = image[:, :, 0]  # 蓝色通道
            
            # 计算红蓝比值作为温度指标
            red_blue_ratio = np.divide(red_channel.astype(float), 
                                     blue_channel.astype(float) + 1e-8)  # 避免除零
            
            temp_ratio = red_blue_ratio[mask]
            
        else:
            # 如果是单通道图像
            gray = image
            mask = gray > 10
            if np.sum(mask) == 0:
                return {}
            temp_proxy = gray[mask]
            temp_ratio = temp_proxy  # 单通道时两者相同
        
        # 基于色调的温度特征
        features['hue_temp_mean'] = np.mean(temp_proxy)
        features['hue_temp_std'] = np.std(temp_proxy)
        features['hue_temp_min'] = np.min(temp_proxy)
        features['hue_temp_max'] = np.max(temp_proxy)
        features['hue_temp_median'] = np.median(temp_proxy)
        features['hue_temp_range'] = features['hue_temp_max'] - features['hue_temp_min']
        
        # 基于红蓝比值的温度特征
        features['rb_ratio_mean'] = np.mean(temp_ratio)
        features['rb_ratio_std'] = np.std(temp_ratio)
        features['rb_ratio_max'] = np.max(temp_ratio)
        features['rb_ratio_min'] = np.min(temp_ratio)
        
        # 百分位数特征
        features['hue_temp_p25'] = np.percentile(temp_proxy, 25)
        features['hue_temp_p75'] = np.percentile(temp_proxy, 75)
        features['hue_temp_iqr'] = features['hue_temp_p75'] - features['hue_temp_p25']
        
        # 偏度和峰度
        from scipy.stats import skew, kurtosis
        features['hue_temp_skewness'] = skew(temp_proxy)
        features['hue_temp_kurtosis'] = kurtosis(temp_proxy)
        
        # 熵
        features['hue_temp_entropy'] = shannon_entropy(temp_proxy.astype(np.uint8))
        
        # 前景区域比例
        features['foreground_ratio'] = np.sum(mask) / mask.size
        
        # 高温区域特征（基于色调）
        if len(image.shape) == 3:
            # 假设色调值0-30和150-180为红色（高温）区域
            high_temp_mask = ((hue >= 0) & (hue <= 30)) | ((hue >= 150) & (hue <= 180))
            high_temp_mask = high_temp_mask & mask  # 结合前景掩码
            features['high_temp_ratio'] = np.sum(high_temp_mask) / np.sum(mask) if np.sum(mask) > 0 else 0
        
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
                    
                    # 边界框特征
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    features[f'bbox_aspect_ratio_{i}'] = w / h if h > 0 else 0
                    features[f'bbox_extent_{i}'] = area / (w * h) if (w * h) > 0 else 0
                    
            except Exception as e:
                print(f"形状特征计算错误: {e}")
        
        return features
    
    def load_and_extract_multimodal_features(self):
        """加载并提取多模态特征"""
        print("开始提取多模态特征...")
        
        # 1. 加载临床特征
        clinical_df = self.load_clinical_features()
        if clinical_df.empty:
            raise ValueError("无法加载临床特征数据")
        
        # 2. 编码临床特征
        clinical_encoded, clinical_feature_names = self.encode_clinical_features(clinical_df)
        
        # 3. 提取图像特征
        all_image_features = []
        all_labels = []
        all_patient_ids = []
        
        # 遍历每个类别
        for class_name in ['icas', 'non_icas']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: 类别目录不存在 {class_dir}")
                continue
            
            class_label = self.class_to_idx[class_name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            print(f"处理类别 {class_name}: {len(image_files)} 张图像")
            
            for img_path in tqdm(image_files, desc=f"提取 {class_name} 图像特征"):
                # 从文件名提取患者ID - 下划线前的字段 - 下划线前的字段
                filename = img_path.stem
                
                # 按下划线分割，取第一部分作为patient_id
                if '_' in filename:
                    patient_id = filename.split('_')[0]
                else:
                    # 如果没有下划线，使用整个文件名（去掉可能的数字后缀）
                    patient_id = filename
                
                #print(f"文件 {filename} -> 患者ID: {patient_id}")  # 调试信息
                
                # 提取图像特征
                image_features = self.extract_image_features(img_path)
                if image_features is not None:
                    all_image_features.append(image_features)
                    all_labels.append(class_label)
                    all_patient_ids.append(patient_id)
        
        # 4. 转换图像特征为DataFrame
        image_features_df = pd.DataFrame(all_image_features)
        image_features_df['patient_id'] = all_patient_ids
        image_features_df['label'] = all_labels
        
        # 5. 合并图像特征和临床特征
        print("合并图像特征和临床特征...")
        
        # 显示提取到的患者ID样例
        unique_image_patients = set(all_patient_ids)
        print(f"从图像文件名提取到的患者ID样例: {list(unique_image_patients)[:10]}")
        
        merged_df = image_features_df.merge(clinical_encoded, on='patient_id', how='left')
        
        # 检查合并结果
        print(f"图像特征样本数: {len(image_features_df)}")
        print(f"临床特征样本数: {len(clinical_encoded)}")
        print(f"合并后样本数: {len(merged_df)}")
        print(f"成功匹配临床特征的样本数: {merged_df[clinical_feature_names[0]].notna().sum()}")
        
        # 显示匹配详情
        clinical_patients = set(clinical_encoded['patient_id'])
        matched_patients = unique_image_patients & clinical_patients
        unmatched_images = unique_image_patients - clinical_patients
        unmatched_clinical = clinical_patients - unique_image_patients
        
        print(f"成功匹配的患者: {len(matched_patients)}")
        print(f"有图像但无临床数据的患者: {len(unmatched_images)}")
        print(f"有临床数据但无图像的患者: {len(unmatched_clinical)}")
        
        if matched_patients:
            print(f"成功匹配的患者ID示例: {list(matched_patients)[:5]}")
        if unmatched_images:
            print(f"无临床数据的患者ID示例: {list(unmatched_images)[:5]}")
        if unmatched_clinical:
            print(f"无图像数据的患者ID示例: {list(unmatched_clinical)[:5]}")
        
        # 处理缺失的临床特征
        for col in clinical_feature_names:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        
        # 使用数据库中的ICAS标签（如果可用）
        if 'has_icas' in merged_df.columns:
            # 对于有临床数据的样本，使用数据库中的标签
            mask = merged_df['has_icas'].notna()
            merged_df.loc[mask, 'label'] = merged_df.loc[mask, 'has_icas'].astype(int)
        
        print(f"最终特征维度: 图像特征 {len(image_features_df.columns)-2} + 临床特征 {len(clinical_feature_names)}")
        print(f"ICAS: {merged_df['label'].sum()} 张")
        print(f"Non-ICAS: {len(merged_df) - merged_df['label'].sum()} 张")
        
        # 保存特征
        features_path = self.run_dir / "multimodal_features.csv"
        merged_df.to_csv(features_path, index=False)
        print(f"多模态特征已保存到: {features_path}")
        
        return merged_df, clinical_feature_names
    
    def prepare_multimodal_data(self, features_df, clinical_feature_names):
        """准备多模态数据 - 改进版本"""
        print("准备多模态数据...")
        
        # 1. 分离特征和标签
        # 排除非特征列：patient_id, label, has_icas
        exclude_cols = ['patient_id', 'label', 'has_icas']
        feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_columns]
        y = features_df['label']  # 使用label作为目标变量
        
        print(f"特征列数: {len(feature_columns)}")
        print(f"样本数: {len(X)}")
        print(f"目标变量分布: {y.value_counts().to_dict()}")
        
        # 分离图像特征和临床特征
        image_feature_cols = [col for col in feature_columns if col not in clinical_feature_names]
        clinical_feature_cols = [col for col in feature_columns if col in clinical_feature_names]
        
        print(f"图像特征维度: {len(image_feature_cols)}")
        print(f"临床特征维度: {len(clinical_feature_cols)}")
        
        # 处理缺失值
        print("处理缺失值...")
        # 图像特征用0填充
        X[image_feature_cols] = X[image_feature_cols].fillna(0)
        # 临床特征用中位数填充
        for col in clinical_feature_cols:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # 数据分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )
        
        # 特征选择 - 分别处理图像和临床特征
        print("进行特征选择...")
        
        # 1. 方差阈值过滤（主要针对图像特征）
        var_threshold = VarianceThreshold(threshold=0.01)
        X_train_image = var_threshold.fit_transform(X_train[image_feature_cols])
        X_val_image = var_threshold.transform(X_val[image_feature_cols])
        X_test_image = var_threshold.transform(X_test[image_feature_cols])
        
        selected_image_features = [image_feature_cols[i] for i in range(len(image_feature_cols)) 
                                 if var_threshold.get_support()[i]]
        
        # 2. 临床特征保留所有（通常维度不高）
        X_train_clinical = X_train[clinical_feature_cols].values
        X_val_clinical = X_val[clinical_feature_cols].values
        X_test_clinical = X_test[clinical_feature_cols].values
        
        # 3. 图像特征降维（可选）
        from sklearn.decomposition import PCA
        if X_train_image.shape[1] > 100:  # 如果图像特征维度过高
            print(f"图像特征维度过高({X_train_image.shape[1]})，进行PCA降维...")
            pca = PCA(n_components=min(100, X_train_image.shape[1]), random_state=42)
            X_train_image = pca.fit_transform(X_train_image)
            X_val_image = pca.transform(X_val_image)
            X_test_image = pca.transform(X_test_image)
            print(f"PCA后图像特征维度: {X_train_image.shape[1]}")
            print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 4. 标准化处理
        print("进行特征标准化...")
        from sklearn.preprocessing import StandardScaler, RobustScaler
        
        # 图像特征标准化
        image_scaler = RobustScaler()  # 对异常值更鲁棒
        X_train_image_scaled = image_scaler.fit_transform(X_train_image)
        X_val_image_scaled = image_scaler.transform(X_val_image)
        X_test_image_scaled = image_scaler.transform(X_test_image)
        
        # 临床特征标准化
        clinical_scaler = StandardScaler()
        X_train_clinical_scaled = clinical_scaler.fit_transform(X_train_clinical)
        X_val_clinical_scaled = clinical_scaler.transform(X_val_clinical)
        X_test_clinical_scaled = clinical_scaler.transform(X_test_clinical)
        
        # 5. 特征融合
        X_train_final = np.hstack([X_train_image_scaled, X_train_clinical_scaled])
        X_val_final = np.hstack([X_val_image_scaled, X_val_clinical_scaled])
        X_test_final = np.hstack([X_test_image_scaled, X_test_clinical_scaled])
        
        print(f"最终特征维度: {X_train_final.shape[1]} (图像: {X_train_image_scaled.shape[1]}, 临床: {X_train_clinical_scaled.shape[1]})")
        print(f"训练集样本数: {len(X_train_final)}")
        print(f"验证集样本数: {len(X_val_final)}")
        print(f"测试集样本数: {len(X_test_final)}")
        
        feature_info = {
            'total_features': X_train_final.shape[1],
            'image_features': X_train_image_scaled.shape[1],
            'clinical_features': X_train_clinical_scaled.shape[1],
            'selected_image_features': selected_image_features,
            'clinical_features_list': clinical_feature_cols
        }
        
        scalers = {
            'image_scaler': image_scaler,
            'clinical_scaler': clinical_scaler,
            'var_threshold': var_threshold
        }
        
        return (X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, 
                feature_info, scalers)
    
    def handle_imbalanced_data(self, X_train, y_train, method='smote'):
        """处理数据不平衡"""
        print(f"原始训练集类别分布: {np.bincount(y_train)}")
        
        if method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"SMOTE后类别分布: {np.bincount(y_train_balanced)}")
                return X_train_balanced, y_train_balanced
            except ImportError:
                print("imblearn未安装，跳过SMOTE处理")
                return X_train, y_train
        
        elif method == 'adasyn':
            try:
                from imblearn.over_sampling import ADASYN
                adasyn = ADASYN(random_state=42)
                X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
                print(f"ADASYN后类别分布: {np.bincount(y_train_balanced)}")
                return X_train_balanced, y_train_balanced
            except ImportError:
                print("imblearn未安装，跳过ADASYN处理")
                return X_train, y_train
        
        else:
            return X_train, y_train
    
    def train_multimodal_models(self, X_train, X_val, y_train, y_val):
        """训练多模态机器学习模型 - 改进版本"""
        print("开始训练多模态机器学习模型...")
        
        # 导入所需的模型
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.neural_network import MLPClassifier
        
        # 计算类别权重
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        print(f"类别权重: {class_weight_dict}")
        print(f"类别分布 - 训练集: {np.bincount(y_train)}")
        print(f"类别分布 - 验证集: {np.bincount(y_val)}")
        
        # 定义改进的基础模型
        base_models = {
            'XGBoost': None,
            'LightGBM': None,
            'CatBoost': None,
            'RandomForest': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            ),
            'MLP_3Layer': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),  # 3层隐藏层
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2正则化
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                shuffle=True,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                tol=1e-4
            ),
            'MLP_Simple': MLPClassifier(
                hidden_layer_sizes=(128, 64),  # 2层隐藏层对比
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # 尝试添加梯度提升模型
        try:
            import xgboost as xgb
            scale_pos_weight = class_weights[0] / class_weights[1] if len(class_weights) > 1 else 1
            base_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        except ImportError:
            print("XGBoost未安装，跳过XGBoost模型")
            del base_models['XGBoost']
        
        try:
            import lightgbm as lgb
            base_models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        except ImportError:
            print("LightGBM未安装，跳过LightGBM模型")
            del base_models['LightGBM']
        
        try:
            import catboost as cb
            base_models['CatBoost'] = cb.CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.1,
                class_weights=list(class_weights),
                random_state=42,
                verbose=False
            )
        except ImportError:
            print("CatBoost未安装，跳过CatBoost模型")
            del base_models['CatBoost']
        
        # 创建模型集合（包括Focal Loss版本）
        models = {}
        for name, base_model in base_models.items():
            if base_model is not None:
                models[name] = base_model
                # 只为部分模型添加Focal Loss版本
                if name in ['XGBoost', 'LightGBM', 'RandomForest']:
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
                val_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    val_pred_proba = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, 'decision_function'):
                    val_pred_proba = model.decision_function(X_val)
                
                # 计算指标
                accuracy = accuracy_score(y_val, val_pred)
                precision = precision_score(y_val, val_pred, zero_division=0)
                recall = recall_score(y_val, val_pred, zero_division=0)
                f1 = f1_score(y_val, val_pred, zero_division=0)
                
                auc = 0
                if val_pred_proba is not None:
                    try:
                        auc = roc_auc_score(y_val, val_pred_proba)
                    except:
                        auc = 0
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'training_time': training_time
                }
                
                trained_models[name] = model
                
                print(f"  准确率: {accuracy:.4f}")
                print(f"  F1分数: {f1:.4f}")
                print(f"  AUC: {auc:.4f}")
                print(f"  训练时间: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"  训练失败: {str(e)}")
                continue
        
        return results, trained_models
    
    def evaluate_best_model(self, trained_models, results, X_test, y_test):
        """评估最佳模型"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 选择F1分数最高的模型
        valid_results = {k: v for k, v in results.items() if v is not None}
        best_model_name = max(valid_results.keys(), 
                            key=lambda x: valid_results[x]['f1'])
        best_model = trained_models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"验证集F1分数: {valid_results[best_model_name]['f1']:.4f}")
        
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
        model_path = self.run_dir / f'best_multimodal_model_{best_model_name}.pkl'
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
    
    def save_results(self, results, test_results, feature_info, config):
        """保存训练结果"""
        # 保存模型比较结果
        results_df = pd.DataFrame(results).T
        results_path = self.run_dir / "multimodal_model_comparison.csv"
        results_df.to_csv(results_path)
        
        # 保存完整结果
        full_results = {
            'config': config,
            'data_info': {
                'data_dir': str(self.data_dir),
                'db_path': str(self.db_path),
                'feature_info': feature_info
            },
            'validation_results': results,
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.run_dir / 'multimodal_full_results.json', 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"所有结果已保存到: {self.run_dir}")
    
    def run_training(self, config=None):
        """运行完整的多模态训练流程 - 改进版本"""
        if config is None:
            config = {
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'random_state': 42,
                'test_size': 0.3,
                'val_size': 0.2,
                'balance_method': 'smote',  # 'smote', 'adasyn', 'none'
                'use_pca': True,
                'pca_components': 100
            }
        
        print("=== 多模态热力图ICAS分类 (图像+临床特征) - 改进版 ===\n")
        
        start_time = time.time()
        
        # 1. 提取多模态特征
        features_df, clinical_feature_names = self.load_and_extract_multimodal_features()
        
        # 2. 准备数据
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         feature_info, scalers) = self.prepare_multimodal_data(features_df, clinical_feature_names)
        
        # 3. 处理数据不平衡（可选）
        if config.get('balance_method', 'none') != 'none':
            X_train, y_train = self.handle_imbalanced_data(
                X_train, y_train, method=config['balance_method']
            )
        
        # 4. 训练模型
        results, trained_models = self.train_multimodal_models(X_train, X_val, y_train, y_val)
        
        # 5. 评估最佳模型
        best_model_name, best_model, test_results = self.evaluate_best_model(
            trained_models, results, X_test, y_test)

        # 创建描述性输出目录
        self.run_dir = self.create_descriptive_output_dir(
            best_model_name, config['focal_loss_alpha'], config['focal_loss_gamma']
        )
        print(f"输出目录: {self.run_dir}")

        total_time = time.time() - start_time
        config['total_training_time'] = total_time

        # 6. 保存结果
        self.save_results(results, test_results, feature_info, config)
        
        print(f"\n=== 多模态训练完成 ===")
        print(f"最佳模型: {best_model_name}")
        print(f"测试准确率: {test_results['test_accuracy']:.4f}")
        print(f"测试F1分数: {test_results['test_f1']:.4f}")
        print(f"测试AUC: {test_results['test_auc']:.4f}")
        print(f"总训练时间: {total_time:.2f}秒")
        
        return results, best_model, test_results

def main():
    # 配置参数
    config = {
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'random_state': 42,
        'test_size': 0.3,
        'val_size': 0.5
    }
    
    # 创建多模态分类器
    classifier = MultiModalThermalClassifier(
        data_dir="./dataset/datasets/thermal_classification_cropped",
        db_path="./web/database/patientcare.db",
        output_dir="./model/multimodal_thermal_classifier_results"
    )
    
    # 运行训练
    results, best_model, test_results = classifier.run_training(config)
    
    print(f"\n=== 最终结果 ===")
    print(f"最佳模型: {test_results['best_model']}")
    print(f"测试准确率: {test_results['test_accuracy']:.4f}")
    print(f"测试F1分数: {test_results['test_f1']:.4f}")
    print(f"测试AUC: {test_results['test_auc']:.4f}")

if __name__ == "__main__":
    main()
