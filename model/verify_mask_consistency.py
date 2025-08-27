#!/usr/bin/env python3
"""
验证训练脚本4中mask使用的一致性
确保对比学习和分类微调阶段都正确使用了mask
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

def check_mask_consistency():
    """检查mask使用的一致性"""
    print("=== 验证Mask使用一致性 ===\n")
    
    # 读取训练脚本内容
    script_path = Path(__file__).parent / "train_thermal_classifier4.py"
    
    if not script_path.exists():
        print("❌ 训练脚本不存在")
        return False
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键配置
    checks = []
    
    # 1. 检查数据集创建是否传递mask参数
    dataset_creation_patterns = [
        "use_face_mask=self.use_face_mask",
        "mask_type=self.mask_type"
    ]
    
    for pattern in dataset_creation_patterns:
        if pattern in content:
            checks.append(f"✅ 数据集创建包含: {pattern}")
        else:
            checks.append(f"❌ 数据集创建缺少: {pattern}")
    
    # 2. 检查对比学习阶段的attention逻辑
    contrastive_attention_pattern = "if self.use_attention:"
    if contrastive_attention_pattern in content:
        # 检查是否有正确的条件判断
        if "if not self.use_face_mask:" in content:
            checks.append("✅ 对比学习阶段有正确的attention逻辑")
        else:
            checks.append("❌ 对比学习阶段attention逻辑可能有问题")
    else:
        checks.append("❌ 对比学习阶段缺少attention逻辑")
    
    # 3. 检查分类训练阶段的attention逻辑
    # 统计attention_mask相关代码出现次数
    attention_mask_count = content.count("attention_mask = None")
    if attention_mask_count >= 3:  # 训练、验证、测试阶段都应该有
        checks.append(f"✅ 分类阶段attention_mask初始化: {attention_mask_count}处")
    else:
        checks.append(f"❌ 分类阶段attention_mask初始化不足: {attention_mask_count}处")
    
    # 4. 检查模型调用是否传递attention_mask
    model_call_patterns = [
        "model(img1, attention_mask1)",
        "model(img2, attention_mask2)", 
        "model.classify(img, attention_mask)"
    ]
    
    for pattern in model_call_patterns:
        if pattern in content:
            checks.append(f"✅ 模型调用包含: {pattern}")
        else:
            checks.append(f"❌ 模型调用缺少: {pattern}")
    
    # 5. 检查默认配置
    if 'mask_type="content_based"' in content:
        checks.append("✅ 默认使用智能mask")
    else:
        checks.append("⚠️  未使用智能mask作为默认")
    
    if 'use_face_mask=True' in content:
        checks.append("✅ 默认启用face_mask")
    else:
        checks.append("❌ 默认未启用face_mask")
    
    # 显示检查结果
    print("检查结果:")
    for check in checks:
        print(f"  {check}")
    
    # 统计结果
    success_count = sum(1 for check in checks if check.startswith("✅"))
    warning_count = sum(1 for check in checks if check.startswith("⚠️"))
    error_count = sum(1 for check in checks if check.startswith("❌"))
    
    print(f"\n总结:")
    print(f"  ✅ 通过: {success_count}")
    print(f"  ⚠️  警告: {warning_count}")
    print(f"  ❌ 错误: {error_count}")
    
    if error_count == 0:
        print("\n🎉 Mask使用一致性检查通过!")
        return True
    else:
        print(f"\n❌ 发现 {error_count} 个问题，需要修复")
        return False

def check_training_flow():
    """检查训练流程的逻辑"""
    print("\n=== 检查训练流程逻辑 ===\n")
    
    print("训练流程中mask的使用:")
    print("1. 📊 数据预处理阶段:")
    print("   - 如果 use_face_mask=True:")
    print("     * 在图像加载时应用content_based mask")
    print("     * 将mask应用到PIL图像，然后进行resize和normalize")
    print("   - 如果 use_face_mask=False:")
    print("     * 直接进行resize和normalize，不应用mask")
    
    print("\n2. 🧠 模型前向传播阶段:")
    print("   - 如果 use_attention=True:")
    print("     * 如果已经使用了face_mask: 不再生成attention_mask")
    print("     * 如果没有使用face_mask: 动态生成attention_mask")
    print("   - 如果 use_attention=False:")
    print("     * 不使用任何attention机制")
    
    print("\n3. 🎯 推荐配置:")
    print("   - use_face_mask=True + mask_type='content_based' + use_attention=True")
    print("   - 这样可以在预处理阶段精确去除背景，同时保留attention机制的灵活性")
    
    print("\n4. ⚠️  注意事项:")
    print("   - 对比学习和分类微调必须使用相同的mask配置")
    print("   - 如果改变mask设置，需要重新训练整个模型")
    print("   - mask的应用会改变输入数据分布，影响模型性能")

def main():
    """主函数"""
    print("Mask一致性验证工具\n")
    
    # 运行检查
    consistency_ok = check_mask_consistency()
    check_training_flow()
    
    if consistency_ok:
        print("\n🎉 验证完成，mask使用一致性良好!")
        print("现在可以安全地运行训练脚本4")
    else:
        print("\n⚠️  发现一致性问题，建议检查代码")
    
    print(f"\n📝 配置建议:")
    print("对于黑色背景的热力图，推荐使用:")
    print("```python")
    print("classifier = ContrastiveThermalClassifier(")
    print("    use_face_mask=True,")
    print("    mask_type='content_based',")
    print("    use_attention=True")
    print(")")
    print("```")

if __name__ == "__main__":
    main()
