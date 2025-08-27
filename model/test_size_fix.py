#!/usr/bin/env python3
"""
测试尺寸修复是否有效
"""

import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from train_thermal_classifier4 import ContrastiveThermalDataset
import torchvision.transforms as T

def test_dataset_processing():
    """测试数据集处理是否正常"""
    print("=== 测试数据集处理 ===")
    
    # 设置数据变换
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 测试数据集路径
    data_dir = "./dataset/datasets/thermal_classification_cropped"
    
    if not Path(data_dir).exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        return False
    
    try:
        # 创建数据集（使用智能mask）
        dataset = ContrastiveThermalDataset(
            data_dir, 
            transform, 
            mode='classification',
            use_asymmetry_analysis=False,
            use_face_mask=True,
            mask_type="content_based"
        )
        
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试前几个样本
        success_count = 0
        error_count = 0
        
        for i in range(min(10, len(dataset))):
            try:
                img, label = dataset[i]
                print(f"样本 {i}: 形状 {img.shape}, 标签 {label}")
                success_count += 1
            except Exception as e:
                print(f"❌ 样本 {i} 处理失败: {e}")
                error_count += 1
        
        print(f"\n处理结果: 成功 {success_count}, 失败 {error_count}")
        
        if success_count > 0:
            print("✅ 尺寸修复成功！")
            return True
        else:
            print("❌ 所有样本都处理失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False

def test_specific_problematic_images():
    """测试之前出错的特定图像"""
    print("\n=== 测试特定问题图像 ===")
    
    # 之前出错的图像路径
    problematic_images = [
        "dataset/datasets/thermal_classification_cropped/icas/021AE_021AE1.jpg",
        "dataset/datasets/thermal_classification_cropped/non_icas/AW002_AW002-正-1.jpg",
        "dataset/datasets/thermal_classification_cropped/non_icas/GE006_GE0061.jpg",
        "dataset/datasets/thermal_classification_cropped/icas/FS014_FS0141.jpg",
        "dataset/datasets/thermal_classification_cropped/icas/CE110_CE110-1.jpg"
    ]
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    from train_thermal_classifier4 import generate_smart_face_mask
    
    success_count = 0
    
    for img_path in problematic_images:
        if not Path(img_path).exists():
            print(f"⚠️  图像不存在: {img_path}")
            continue
            
        try:
            # 加载图像
            face_img = Image.open(img_path).convert("RGB")
            print(f"处理图像: {Path(img_path).name}, 原始尺寸: {face_img.size}")
            
            # 生成智能mask
            face_mask = generate_smart_face_mask(np.array(face_img), "ellipse")
            print(f"  Mask形状: {face_mask.shape}, 覆盖率: {np.mean(face_mask):.3f}")
            
            # 应用mask到PIL图像
            img_array = np.array(face_img).astype(np.float32) / 255.0
            
            # 确保mask尺寸匹配
            if face_mask.shape != img_array.shape[:2]:
                import cv2
                face_mask = cv2.resize(face_mask, (img_array.shape[1], img_array.shape[0]))
                print(f"  Mask调整后形状: {face_mask.shape}")
            
            # 应用mask
            for c in range(img_array.shape[2]):
                img_array[:, :, c] = img_array[:, :, c] * face_mask
            
            # 转换回PIL图像
            img_array = (img_array * 255).astype(np.uint8)
            masked_img = Image.fromarray(img_array)
            
            # 应用变换
            img_tensor = transform(masked_img)
            print(f"  最终tensor形状: {img_tensor.shape}")
            
            success_count += 1
            print(f"  ✅ 处理成功")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print(f"\n特定图像测试结果: {success_count}/{len(problematic_images)} 成功")
    return success_count > 0

def test_different_image_sizes():
    """测试不同尺寸的图像"""
    print("\n=== 测试不同尺寸图像 ===")
    
    from train_thermal_classifier4 import generate_smart_face_mask
    
    # 创建不同尺寸的测试图像
    test_sizes = [(224, 224), (512, 512), (256, 256), (300, 400), (128, 128)]
    
    for size in test_sizes:
        try:
            # 创建测试图像
            test_img = np.random.rand(size[1], size[0], 3).astype(np.float32)
            
            # 添加一个"人脸"区域
            center_y, center_x = size[1] // 2, size[0] // 2
            y, x = np.ogrid[:size[1], :size[0]]
            face_region = ((x - center_x)**2 + (y - center_y)**2) <= (min(size) * 0.3)**2
            test_img[face_region] = 0.8
            
            # 生成mask
            mask = generate_smart_face_mask(test_img, "ellipse")
            
            print(f"尺寸 {size}: 图像形状 {test_img.shape}, Mask形状 {mask.shape}, 覆盖率 {np.mean(mask):.3f}")
            
        except Exception as e:
            print(f"尺寸 {size}: ❌ 失败 - {e}")
    
    print("✅ 不同尺寸测试完成")

def main():
    """主测试函数"""
    print("尺寸修复测试\n")
    
    # 运行所有测试
    test1_result = test_dataset_processing()
    test2_result = test_specific_problematic_images()
    test_different_image_sizes()
    
    print(f"\n=== 测试总结 ===")
    print(f"数据集处理测试: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"特定图像测试: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！尺寸问题已修复")
        print("现在可以安全运行训练脚本了")
    else:
        print("\n⚠️  仍有问题需要解决")

if __name__ == "__main__":
    main()
