import os
import shutil
from pathlib import Path
import random

class YOLODatasetBuilder:
    def __init__(self, source_dir='./dataset/datasets/thermal_24h/icas', target_dir='./dataset/datasets/thermal_24h_yolo', train_ratio=0.8):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        
        # 创建YOLO数据集目录结构
        self.images_train_dir = self.target_dir / 'images' / 'train'
        self.images_val_dir = self.target_dir / 'images' / 'val'
        self.labels_train_dir = self.target_dir / 'labels' / 'train'
        self.labels_val_dir = self.target_dir / 'labels' / 'val'
        
        self._create_directories()
    
    def _create_directories(self):
        """创建YOLO数据集目录结构"""
        for dir_path in [self.images_train_dir, self.images_val_dir, 
                        self.labels_train_dir, self.labels_val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {dir_path}")
    
    def find_image_label_pairs(self):
        """查找图像和标签文件对"""
        pairs = []
        
        if not self.source_dir.exists():
            print(f"错误: 源目录不存在 - {self.source_dir}")
            return pairs
        
        # 查找所有jpg文件
        jpg_files = list(self.source_dir.glob('*.jpg'))
        print(f"找到 {len(jpg_files)} 个jpg文件")
        
        for jpg_file in jpg_files:
            # 查找对应的txt文件
            txt_file = jpg_file.with_suffix('.txt')
            
            if txt_file.exists():
                pairs.append({
                    'image': jpg_file,
                    'label': txt_file,
                    'basename': jpg_file.stem
                })
                print(f"找到配对: {jpg_file.name} <-> {txt_file.name}")
            else:
                print(f"警告: 找不到对应的标签文件 {txt_file.name}")
        
        print(f"总共找到 {len(pairs)} 个有效的图像-标签对")
        return pairs
    
    def split_train_val(self, pairs):
        """将数据分割为训练集和验证集"""
        # 随机打乱数据
        random.shuffle(pairs)
        
        # 计算分割点
        train_count = int(len(pairs) * self.train_ratio)
        
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:]
        
        print(f"数据分割: 训练集 {len(train_pairs)} 个, 验证集 {len(val_pairs)} 个")
        return train_pairs, val_pairs
    
    def copy_files(self, pairs, images_dir, labels_dir, split_name):
        """复制文件到目标目录"""
        success_count = 0
        
        for pair in pairs:
            try:
                # 复制图像文件
                target_image = images_dir / pair['image'].name
                shutil.copy2(pair['image'], target_image)
                
                # 复制标签文件
                target_label = labels_dir / pair['label'].name
                shutil.copy2(pair['label'], target_label)
                
                success_count += 1
                print(f"[{split_name}] 复制成功: {pair['basename']}")
                
            except Exception as e:
                print(f"[{split_name}] 复制失败 {pair['basename']}: {e}")
        
        print(f"[{split_name}] 成功复制 {success_count}/{len(pairs)} 个文件对")
        return success_count
    
    def create_dataset_yaml(self):
        """创建YOLO数据集配置文件"""
        yaml_content = f"""# YOLO数据集配置文件
path: {self.target_dir.absolute()}  # 数据集根目录
train: images/train  # 训练图像目录 (相对于path)
val: images/val      # 验证图像目录 (相对于path)

# 类别
nc: 1  # 类别数量
names: ['target']  # 类别名称
"""
        
        yaml_path = self.target_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"创建配置文件: {yaml_path}")
        return yaml_path
    
    def build_dataset(self):
        """构建YOLO数据集"""
        print("开始构建YOLO分割数据集...")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
        print(f"训练/验证比例: {self.train_ratio:.1%}/{1-self.train_ratio:.1%}")
        
        # 1. 查找图像-标签对
        pairs = self.find_image_label_pairs()
        
        if not pairs:
            print("错误: 没有找到有效的图像-标签对")
            return
        
        # 2. 分割训练集和验证集
        train_pairs, val_pairs = self.split_train_val(pairs)
        
        # 3. 复制训练集文件
        train_success = self.copy_files(train_pairs, self.images_train_dir, 
                                       self.labels_train_dir, "训练集")
        
        # 4. 复制验证集文件
        val_success = self.copy_files(val_pairs, self.images_val_dir, 
                                     self.labels_val_dir, "验证集")
        
        # 5. 创建数据集配置文件
        yaml_path = self.create_dataset_yaml()
        
        # 6. 打印总结
        print(f"\n=== YOLO数据集构建完成 ===")
        print(f"总文件对数: {len(pairs)}")
        print(f"训练集: {train_success} 个文件对")
        print(f"验证集: {val_success} 个文件对")
        print(f"配置文件: {yaml_path}")
        print(f"数据集目录: {self.target_dir}")
        
        return {
            'total_pairs': len(pairs),
            'train_count': train_success,
            'val_count': val_success,
            'yaml_path': yaml_path,
            'dataset_dir': self.target_dir
        }

if __name__ == "__main__":
    # 设置随机种子以确保可重现的分割
    random.seed(42)
    
    builder = YOLODatasetBuilder(
        source_dir='./dataset/datasets/thermal_24h/icas',
        target_dir='./dataset/datasets/thermal_24h_yolo',
        train_ratio=0.8
    )
    
    result = builder.build_dataset()