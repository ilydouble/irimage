import cv2
import numpy as np
import os
from pathlib import Path
import shutil

class SegmentedRegionExtractor:
    def __init__(self, base_dir="./dataset/datasets"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "thermal_classification"
        
        # 定义输入目录映射：原图目录 -> 预测结果目录
        self.input_mapping = {
            'icas': [
                {
                    'source_dir': self.base_dir / "thermal_24h" / "icas",
                    'result_dir': self.base_dir / "thermal_24h" / "icas_result" / "predictions"
                },
                {
                    'source_dir': self.base_dir / "thermal_25h" / "icas", 
                    'result_dir': self.base_dir / "thermal_25h" / "icas_result" / "predictions"
                }
            ],
            'non_icas': [
                {
                    'source_dir': self.base_dir / "thermal_24h" / "non_icas",
                    'result_dir': self.base_dir / "thermal_24h" / "non_icas_result" / "predictions"
                },
                {
                    'source_dir': self.base_dir / "thermal_25h" / "non_icas",
                    'result_dir': self.base_dir / "thermal_25h" / "non_icas_result" / "predictions"
                }
            ]
        }
        
        # 创建输出目录
        self.setup_output_dirs()
        
    def setup_output_dirs(self):
        """创建输出目录结构"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "icas").mkdir(exist_ok=True)
        (self.output_dir / "non_icas").mkdir(exist_ok=True)
        print(f"输出目录创建完成: {self.output_dir}")
    
    def parse_yolo_segmentation(self, label_path, img_width, img_height):
        """解析YOLO分割标签文件"""
        if not label_path.exists():
            print(f"    标签文件不存在: {label_path}")
            return []
        
        masks = []
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"    标签文件为空: {label_path}")
                return []
            
            print(f"    标签文件: {label_path.name}")
            print(f"    图像尺寸: {img_width}x{img_height}")
            
            for line_num, line in enumerate(content.split('\n'), 1):
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                print(f"    行 {line_num}: {parts[:5]}... (共{len(parts)}个)")
                
                if len(parts) < 7:  # class_id + 至少3个点(6个坐标)
                    print(f"    跳过: 数据不足")
                    continue
                
                # 简单假设格式: class_id x1 y1 x2 y2 x3 y3 ...
                try:
                    coords = [float(x) for x in parts[1:]]  # 跳过class_id
                    print(f"    原始坐标数量: {len(coords)}")
                    
                    # 确保是偶数个坐标
                    if len(coords) % 2 != 0:
                        coords = coords[:-1]
                    
                    # 转换为像素坐标并创建点列表
                    points = []
                    for i in range(0, len(coords), 2):
                        x = coords[i] * img_width
                        y = coords[i+1] * img_height
                        points.append([int(x), int(y)])
                    
                    print(f"    转换后点数: {len(points)}")
                    print(f"    前3个点: {points[:3]}")
                    
                    if len(points) >= 3:
                        masks.append(np.array(points, dtype=np.int32))
                        print(f"    ✓ 成功添加多边形")
                    else:
                        print(f"    ✗ 点数不足")
                        
                except Exception as e:
                    print(f"    解析错误: {e}")
        
        print(f"    最终多边形数量: {len(masks)}")
        return masks
    
    def create_mask_from_polygons(self, polygons, img_shape):
        """从多边形创建掩码"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for polygon in polygons:
            cv2.fillPoly(mask, [polygon], 255)
        
        return mask
    
    def apply_mask_to_image(self, image, mask):
        """将掩码应用到图像，背景涂黑"""
        # 创建三通道掩码
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # 应用掩码：保留掩码区域，其他区域涂黑
        masked_image = image * mask_3ch
        
        return masked_image.astype(np.uint8)
    
    def find_original_image(self, source_dir, base_name):
        """在原图目录中查找对应的图像"""
        # 尝试不同的扩展名
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in extensions:
            img_path = source_dir / f"{base_name}{ext}"
            if img_path.exists():
                return img_path
        
        return None
    
    def find_prediction_label(self, result_dir, base_name):
        """在预测结果目录中查找对应的标签文件"""
        # 先检查predictions子目录下的labels
        labels_dir = result_dir / "predictions" / "labels"
        if labels_dir.exists():
            label_path = labels_dir / f"{base_name}.txt"
            if label_path.exists():
                return label_path
        
        # 再检查直接的labels目录
        labels_dir = result_dir / "labels"
        if labels_dir.exists():
            label_path = labels_dir / f"{base_name}.txt"
            if label_path.exists():
                return label_path
        
        return None
    
    def process_single_mapping(self, source_dir, result_dir, category):
        """处理单个目录映射"""
        print(f"处理映射: {source_dir} -> {result_dir}")
        
        # 检查目录是否存在
        if not source_dir.exists():
            print(f"  原图目录不存在，跳过: {source_dir}")
            return 0
        
        if not result_dir.exists():
            print(f"  预测结果目录不存在，跳过: {result_dir}")
            return 0
        
        processed_count = 0
        
        # 遍历原图目录中的所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            for image_path in source_dir.glob(f"*{ext}"):
                try:
                    base_name = image_path.stem
                    
                    # 查找对应的预测标签
                    label_path = self.find_prediction_label(result_dir, base_name)
                    
                    # 读取原始图像
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"  无法读取图像: {image_path}")
                        continue
                    
                    img_height, img_width = image.shape[:2]
                    
                    # 解析分割标签
                    if label_path and label_path.exists():
                        polygons = self.parse_yolo_segmentation(label_path, img_width, img_height)
                    else:
                        polygons = []
                    
                    if not polygons:
                        print(f"  无有效分割区域: {base_name}")
                        # 如果没有检测到区域，创建全黑图像
                        masked_image = np.zeros_like(image)
                    else:
                        # 创建掩码
                        mask = self.create_mask_from_polygons(polygons, image.shape)
                        
                        # 应用掩码
                        masked_image = self.apply_mask_to_image(image, mask)
                    
                    # 保存结果
                    output_path = self.output_dir / category / image_path.name
                    cv2.imwrite(str(output_path), masked_image)
                    
                    processed_count += 1
                    print(f"  处理完成: {image_path.name} -> {category}/{image_path.name}")
                    
                except Exception as e:
                    print(f"  处理失败 {image_path.name}: {e}")
        
        return processed_count
    
    def extract_all_regions(self):
        """提取所有分割区域"""
        print("开始提取分割区域...")
        
        total_processed = 0
        
        for category, mappings in self.input_mapping.items():
            print(f"\n处理类别: {category}")
            category_count = 0
            
            for mapping in mappings:
                count = self.process_single_mapping(
                    mapping['source_dir'], 
                    mapping['result_dir'], 
                    category
                )
                category_count += count
                total_processed += count
            
            print(f"类别 {category} 处理完成: {category_count} 张图像")
        
        print(f"\n=== 提取完成 ===")
        print(f"总处理图像: {total_processed} 张")
        print(f"输出目录: {self.output_dir}")
        
        # 统计最终结果
        icas_count = len(list((self.output_dir / "icas").glob("*.jpg")))
        non_icas_count = len(list((self.output_dir / "non_icas").glob("*.jpg")))
        
        print(f"最终结果:")
        print(f"  ICAS: {icas_count} 张")
        print(f"  Non-ICAS: {non_icas_count} 张")
        
        return {
            'total': total_processed,
            'icas': icas_count,
            'non_icas': non_icas_count
        }
    
    def visualize_sample(self, category="icas", num_samples=3):
        """可视化一些样本结果"""
        sample_dir = self.output_dir / category
        if not sample_dir.exists():
            print(f"样本目录不存在: {sample_dir}")
            return
        
        images = list(sample_dir.glob("*.jpg"))[:num_samples]
        
        print(f"\n显示 {category} 类别的 {len(images)} 个样本:")
        for img_path in images:
            print(f"  {img_path.name}")
            # 这里可以添加图像显示代码
            # cv2.imshow(f"{category} - {img_path.name}", cv2.imread(str(img_path)))
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def main():
    # 初始化提取器
    extractor = SegmentedRegionExtractor()
    
    # 提取所有分割区域
    results = extractor.extract_all_regions()
    
    # 可视化一些样本（可选）
    # extractor.visualize_sample("icas", 3)
    # extractor.visualize_sample("non_icas", 3)
    
    print(f"\n脚本执行完成!")
    print(f"分类数据集已准备完成，可用于后续的分类模型训练")

if __name__ == "__main__":
    main()
