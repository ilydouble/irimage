#!/usr/bin/env python3
"""
测试人脸mask和attention功能
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from train_thermal_classifier4 import (
    generate_face_mask, 
    apply_mask_to_image, 
    create_attention_mask,
    ThermalEncoder
)

def test_mask_generation():
    """测试mask生成功能"""
    print("=== 测试Mask生成功能 ===")
    
    # 测试不同尺寸和类型的mask
    test_sizes = [(224, 224), (224, 112), (256, 256)]
    mask_types = ["ellipse", "rectangle", "adaptive"]
    
    fig, axes = plt.subplots(len(test_sizes), len(mask_types), figsize=(12, 10))
    
    for i, size in enumerate(test_sizes):
        for j, mask_type in enumerate(mask_types):
            mask = generate_face_mask(size, mask_type)
            
            ax = axes[i, j] if len(test_sizes) > 1 else axes[j]
            ax.imshow(mask, cmap='gray')
            ax.set_title(f'{mask_type}\n{size[0]}x{size[1]}')
            ax.axis('off')
            
            # 打印mask统计信息
            coverage = np.mean(mask)
            print(f"{mask_type} {size}: 覆盖率 {coverage:.3f}")
    
    plt.tight_layout()
    plt.savefig('mask_generation_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Mask生成测试完成")

def test_mask_application():
    """测试mask应用功能"""
    print("\n=== 测试Mask应用功能 ===")
    
    # 创建测试图像 (模拟热力图)
    test_image = np.random.rand(3, 224, 224).astype(np.float32)
    
    # 添加一些"人脸"特征 (中心区域更亮)
    center_y, center_x = 112, 112
    y, x = np.ogrid[:224, :224]
    face_region = ((x - center_x)**2 + (y - center_y)**2) < 80**2
    
    for c in range(3):
        test_image[c][face_region] += 0.5
    
    # 测试不同mask类型
    mask_types = ["ellipse", "rectangle", "adaptive"]
    
    fig, axes = plt.subplots(2, len(mask_types) + 1, figsize=(15, 8))
    
    # 显示原图
    axes[0, 0].imshow(np.transpose(test_image, (1, 2, 0)))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')  # 空白
    
    for i, mask_type in enumerate(mask_types):
        # 生成mask
        mask = generate_face_mask((224, 224), mask_type)
        
        # 应用mask
        masked_image = apply_mask_to_image(test_image, mask, background_value=0.0)
        
        # 显示mask
        axes[0, i+1].imshow(mask, cmap='gray')
        axes[0, i+1].set_title(f'{mask_type} Mask')
        axes[0, i+1].axis('off')
        
        # 显示应用mask后的图像
        axes[1, i+1].imshow(np.transpose(masked_image, (1, 2, 0)))
        axes[1, i+1].set_title(f'应用{mask_type}后')
        axes[1, i+1].axis('off')
        
        # 计算保留的信息量
        original_energy = np.sum(test_image**2)
        masked_energy = np.sum(masked_image**2)
        retention_ratio = masked_energy / original_energy
        print(f"{mask_type}: 信息保留率 {retention_ratio:.3f}")
    
    plt.tight_layout()
    plt.savefig('mask_application_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Mask应用测试完成")

def test_attention_mechanism():
    """测试attention机制"""
    print("\n=== 测试Attention机制 ===")
    
    # 创建模型
    model = ThermalEncoder(backbone='resnet18', use_attention=True)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"输入形状: {test_input.shape}")
    
    # 测试不同的attention mask
    mask_types = ["ellipse", "rectangle", "adaptive"]
    
    with torch.no_grad():
        for mask_type in mask_types:
            # 创建attention mask
            attention_mask = create_attention_mask(test_input, mask_type)
            print(f"{mask_type} attention mask形状: {attention_mask.shape}")
            
            # 前向传播
            features = model.forward(test_input, attention_mask, return_features=True)
            print(f"{mask_type} 输出特征形状: {features.shape}")
            
            # 分类输出
            classification_output = model.classify(test_input, attention_mask)
            print(f"{mask_type} 分类输出形状: {classification_output.shape}")
            
            # 检查输出是否合理
            assert not torch.isnan(features).any(), f"{mask_type} 特征包含NaN"
            assert not torch.isnan(classification_output).any(), f"{mask_type} 分类输出包含NaN"
            
            print(f"✅ {mask_type} attention测试通过")
    
    print("✅ Attention机制测试完成")

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n=== 测试模型兼容性 ===")
    
    # 测试不同配置的模型
    configs = [
        {"use_attention": True, "name": "带Attention"},
        {"use_attention": False, "name": "不带Attention"}
    ]
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        
        model = ThermalEncoder(backbone='resnet18', use_attention=config['use_attention'])
        model.eval()
        
        # 测试输入
        test_input = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            # 测试对比学习模式
            contrastive_output = model.forward(test_input)
            print(f"  对比学习输出形状: {contrastive_output.shape}")
            
            # 测试分类模式
            if config['use_attention']:
                attention_mask = create_attention_mask(test_input, "ellipse")
                classification_output = model.classify(test_input, attention_mask)
            else:
                classification_output = model.classify(test_input)
            
            print(f"  分类输出形状: {classification_output.shape}")
            
            # 验证输出
            assert contrastive_output.shape == (2, 512), "对比学习输出形状错误"
            assert classification_output.shape == (2, 2), "分类输出形状错误"
            
            print(f"  ✅ {config['name']}配置测试通过")
    
    print("✅ 模型兼容性测试完成")

def test_performance_impact():
    """测试性能影响"""
    print("\n=== 测试性能影响 ===")
    
    import time
    
    # 创建测试数据
    test_input = torch.randn(8, 3, 224, 224)  # 较大的batch
    
    configs = [
        {"use_attention": False, "use_mask": False, "name": "基线"},
        {"use_attention": True, "use_mask": False, "name": "仅Attention"},
        {"use_attention": False, "use_mask": True, "name": "仅Mask"},
        {"use_attention": True, "use_mask": True, "name": "Attention+Mask"}
    ]
    
    for config in configs:
        model = ThermalEncoder(backbone='resnet18', use_attention=config['use_attention'])
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                if config['use_attention']:
                    attention_mask = create_attention_mask(test_input, "ellipse") if config['use_mask'] else None
                    _ = model.classify(test_input, attention_mask)
                else:
                    _ = model.classify(test_input)
        
        # 计时
        start_time = time.time()
        num_runs = 20
        
        with torch.no_grad():
            for _ in range(num_runs):
                if config['use_attention']:
                    attention_mask = create_attention_mask(test_input, "ellipse") if config['use_mask'] else None
                    _ = model.classify(test_input, attention_mask)
                else:
                    _ = model.classify(test_input)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
        
        print(f"{config['name']}: 平均推理时间 {avg_time:.2f} ms")
    
    print("✅ 性能影响测试完成")

def main():
    """主测试函数"""
    print("人脸Mask和Attention功能测试\n")
    
    try:
        # 运行所有测试
        test_mask_generation()
        test_mask_application()
        test_attention_mechanism()
        test_model_compatibility()
        test_performance_impact()
        
        print("\n🎉 所有测试通过!")
        print("📁 可视化结果已保存:")
        print("  - mask_generation_test.png")
        print("  - mask_application_test.png")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
