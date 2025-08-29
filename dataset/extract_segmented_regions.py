import cv2
import numpy as np
import os
from pathlib import Path
import shutil

class SegmentedRegionExtractor:
    def __init__(self, base_dir="./dataset/datasets"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "thermal_classification_cropped"
        
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
    
    def get_bounding_box_from_mask(self, mask):
        """从掩码获取最小外接矩形"""
        # 找到所有非零像素的坐标
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            return None
        
        # 计算边界框
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return x_min, y_min, x_max, y_max
    
    def crop_foreground_region(self, image, mask, padding=20):
        """截取前景区域，其他区域填充黑色"""
        # 获取前景的边界框
        bbox = self.get_bounding_box_from_mask(mask)
        
        if bbox is None:
            print("    无法获取边界框，返回全黑图像")
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        x_min, y_min, x_max, y_max = bbox
        
        # 添加padding
        h, w = image.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        print(f"    边界框: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
        
        # 截取区域
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        
        if crop_width <= 0 or crop_height <= 0:
            print("    边界框无效，返回全黑图像")
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 创建裁剪后的图像和掩码
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        
        # 创建三通道掩码
        mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # 应用掩码：保留掩码区域，其他区域涂黑
        masked_image = cropped_image * mask_3ch
        
        print(f"    裁剪后尺寸: {masked_image.shape}")
        
        return masked_image.astype(np.uint8)
    
    def resize_to_square(self, image, target_size=512):
        """将图像调整为正方形，保持长宽比"""
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建正方形画布（黑色背景）
        square_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 将缩放后的图像放置在画布中心
        square_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return square_image
    
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
                        cropped_image = np.zeros((512, 512, 3), dtype=np.uint8)
                    else:
                        # 创建掩码
                        mask = self.create_mask_from_polygons(polygons, image.shape)
                        
                        # 截取前景区域
                        cropped_image = self.crop_foreground_region(image, mask, padding=20)
                        
                        # 调整为正方形
                        cropped_image = self.resize_to_square(cropped_image, target_size=512)
                    
                    # 保存结果
                    output_path = self.output_dir / category / image_path.name
                    cv2.imwrite(str(output_path), cropped_image)
                    
                    processed_count += 1
                    print(f"  处理完成: {image_path.name} -> {category}/{image_path.name}")
                    
                except Exception as e:
                    print(f"  处理失败 {image_path.name}: {e}")
        
        return processed_count
    
    def extract_all_regions(self):
        """提取所有分割区域"""
        print("开始提取分割区域（截取模式）...")
        
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

def main():
    # 初始化提取器
    extractor = SegmentedRegionExtractor()
    
    # 提取所有分割区域
    results = extractor.extract_all_regions()
    
    print(f"\n脚本执行完成!")
    print(f"裁剪后的分类数据集已准备完成，可用于后续的分类模型训练")
    print(f"数据集位置: {extractor.output_dir}")

if __name__ == "__main__":
    main()