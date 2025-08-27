#!/usr/bin/env python3
"""
可视化智能mask效果的简单脚本
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from train_thermal_classifier4 import generate_smart_face_mask, generate_face_mask

def visualize_mask_comparison(image_path: str):
    """可视化不同mask方法的对比效果"""
    
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image) / 255.0
        
        print(f"处理图像: {image_path}")
        print(f"图像尺寸: {image.size}")
        
        # 不同的mask方法
        mask_methods = [
            ("原图", None),
            ("智能检测", "content_based"),
            ("椭圆形", "ellipse"),
            ("矩形", "rectangle"),
            ("自适应", "adaptive")
        ]
        
        fig, axes = plt.subplots(2, len(mask_methods), figsize=(20, 8))
        
        for i, (method_name, mask_type) in enumerate(mask_methods):
            if mask_type is None:
                # 显示原图
                axes[0, i].imshow(image_np)
                axes[1, i].imshow(image_np)
                axes[0, i].set_title(f'{method_name}')
                axes[1, i].set_title('原图')
            else:
                # 生成mask
                if mask_type == "content_based":
                    mask = generate_smart_face_mask(image_np, "ellipse")
                else:
                    mask = generate_face_mask(image.size, mask_type)
                
                # 显示mask
                axes[0, i].imshow(mask, cmap='gray')
                axes[0, i].set_title(f'{method_name} Mask')
                
                # 应用mask到图像
                masked_image = image_np.copy()
                for c in range(3):
                    masked_image[:, :, c] = masked_image[:, :, c] * mask
                
                axes[1, i].imshow(masked_image)
                coverage = np.mean(mask)
                axes[1, i].set_title(f'应用后\n覆盖率: {coverage:.3f}')
                
                print(f"  {method_name}: 覆盖率 {coverage:.3f}")
            
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        output_name = f"mask_comparison_{Path(image_path).stem}.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"结果已保存: {output_name}")
        
        plt.show()
        
    except Exception as e:
        print(f"处理图像时出错: {e}")

def main():
    """主函数"""
    print("智能Mask可视化工具\n")
    
    # 查找测试图像
    test_dirs = [
        "./dataset/datasets/thermal_classification_cropped/icas",
        "./dataset/datasets/thermal_classification_cropped/non_icas",
        "./dataset/datasets/thermal_24h"
    ]
    
    test_images = []
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            images = list(test_path.glob("*.jpg"))[:2]  # 每个目录最多2张
            test_images.extend(images)
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("请确保以下目录存在并包含图像文件:")
        for test_dir in test_dirs:
            print(f"  - {test_dir}")
        return
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 处理每张图像
    for i, image_path in enumerate(test_images[:3]):  # 最多处理3张
        print(f"\n=== 处理第 {i+1} 张图像 ===")
        visualize_mask_comparison(str(image_path))
    
    print(f"\n🎉 可视化完成!")
    print("📁 结果文件已保存为 mask_comparison_*.png")
    
    print(f"\n💡 观察要点:")
    print("1. 智能检测mask应该更贴合实际人脸轮廓")
    print("2. 对于黑色背景的热力图，智能检测效果最佳")
    print("3. 覆盖率应该在0.2-0.6之间比较合理")
    print("4. mask边缘应该相对平滑，没有明显噪声")

if __name__ == "__main__":
    main()
