#!/usr/bin/env python3
"""
热力图分类器重命名脚本
将原有的 train_thermal_classifier*.py 重命名为更具描述性的名称
"""

import os
import shutil
from pathlib import Path

def rename_thermal_classifiers():
    """重命名热力图分类器脚本"""
    
    # 定义重命名映射
    rename_mapping = {
        'train_thermal_classifier.py': 'train_cnn_classifier.py',
        'train_thermal_classifier1.py': 'train_feature_ml_classifier.py', 
        'train_thermal_classifier2.py': 'train_multimodal_classifier.py',
        'train_thermal_classifier3.py': 'train_contrastive_classifier.py',
        'train_thermal_classifier4.py': 'train_contrastive_mask_classifier.py',
        'train_thermal_classifier5.py': 'train_contrastive_split_classifier.py'
    }
    
    model_dir = Path('model')
    backup_dir = model_dir / 'backup_original_classifiers'
    
    print("🔄 开始重命名热力图分类器脚本...")
    print("=" * 60)
    
    # 创建备份目录
    backup_dir.mkdir(exist_ok=True)
    print(f"📁 备份目录: {backup_dir}")
    
    renamed_files = []
    skipped_files = []
    
    for old_name, new_name in rename_mapping.items():
        old_path = model_dir / old_name
        new_path = model_dir / new_name
        backup_path = backup_dir / old_name
        
        if old_path.exists():
            try:
                # 1. 备份原文件
                shutil.copy2(old_path, backup_path)
                print(f"💾 备份: {old_name} -> backup_original_classifiers/{old_name}")
                
                # 2. 重命名文件
                old_path.rename(new_path)
                print(f"✅ 重命名: {old_name} -> {new_name}")
                
                renamed_files.append((old_name, new_name))
                
            except Exception as e:
                print(f"❌ 重命名失败 {old_name}: {e}")
                skipped_files.append(old_name)
        else:
            print(f"⚠️  文件不存在: {old_name}")
            skipped_files.append(old_name)
    
    print("\n" + "=" * 60)
    print("📊 重命名结果汇总:")
    print(f"✅ 成功重命名: {len(renamed_files)} 个文件")
    print(f"⚠️  跳过文件: {len(skipped_files)} 个文件")
    
    if renamed_files:
        print("\n🎉 成功重命名的文件:")
        for old_name, new_name in renamed_files:
            print(f"  • {old_name} → {new_name}")
    
    if skipped_files:
        print("\n⚠️  跳过的文件:")
        for file_name in skipped_files:
            print(f"  • {file_name}")
    
    print(f"\n💾 原始文件已备份到: {backup_dir}")
    print("🔧 如需恢复，可从备份目录复制回来")

def create_classifier_summary():
    """创建分类器功能总结文档"""
    
    summary_content = """# 热力图分类器功能总结

## 📋 脚本重命名对照表

| 原名称 | 新名称 | 核心技术 | 主要特点 |
|--------|--------|----------|----------|
| `train_thermal_classifier.py` | `train_cnn_classifier.py` | **深度学习CNN** | ResNet/EfficientNet + Focal Loss + YOLO11特征提取器 |
| `train_thermal_classifier1.py` | `train_feature_ml_classifier.py` | **传统机器学习** | 手工特征提取 + 多种ML算法 + Focal Loss包装器 |
| `train_thermal_classifier2.py` | `train_multimodal_classifier.py` | **多模态融合** | 图像特征 + 临床数据 + 机器学习 |
| `train_thermal_classifier3.py` | `train_contrastive_classifier.py` | **对比学习** | 两阶段训练：对比学习 + 分类微调 |
| `train_thermal_classifier4.py` | `train_contrastive_mask_classifier.py` | **对比学习 + Mask** | 对比学习 + 智能人脸Mask + Attention机制 |
| `train_thermal_classifier5.py` | `train_contrastive_split_classifier.py` | **对比学习 + 数据分割** | 对比学习 + 改进的数据集分割策略 |

## 🎯 使用场景建议

### 1. **标准深度学习分类** → `train_cnn_classifier.py`
- **适用**: 常规CNN分类任务
- **优势**: 成熟稳定，支持多种预训练模型
- **推荐**: 作为基线模型使用

### 2. **传统机器学习对比** → `train_feature_ml_classifier.py`
- **适用**: 特征工程研究，算法对比
- **优势**: 可解释性强，训练快速
- **推荐**: 特征分析和快速原型

### 3. **临床应用** → `train_multimodal_classifier.py`
- **适用**: 结合患者临床数据的诊断
- **优势**: 信息全面，符合临床实际
- **推荐**: 实际临床部署使用

### 4. **自监督学习** → `train_contrastive_classifier.py`
- **适用**: 数据量大，需要学习表征
- **优势**: 无需大量标注，泛化能力强
- **推荐**: 大规模数据集训练

### 5. **精确人脸分析** → `train_contrastive_mask_classifier.py`
- **适用**: 需要关注人脸特定区域
- **优势**: 可解释性强，精确定位
- **推荐**: 研究人脸热力图模式

### 6. **实验数据控制** → `train_contrastive_split_classifier.py`
- **适用**: 需要精确控制数据分割的实验
- **优势**: 数据分割一致性，实验可重复
- **推荐**: 科研实验和方法对比

## 📈 性能对比参考

根据已有测试结果：

1. **最佳性能**: `train_multimodal_classifier.py` (74.83% 准确率)
2. **最快训练**: `train_feature_ml_classifier.py` (28秒)
3. **最强泛化**: `train_contrastive_classifier.py` (理论上)
4. **最可解释**: `train_contrastive_mask_classifier.py`

## 🔧 迁移指南

如果你之前使用旧名称的脚本：

1. **检查导入**: 更新任何导入这些脚本的代码
2. **更新文档**: 修改相关文档中的脚本名称
3. **配置文件**: 更新配置文件中的脚本路径
4. **备份恢复**: 如需恢复，从 `backup_original_classifiers/` 目录复制

## 📚 相关文档

- [多模态特征文档](docs/multimodal_features_documentation.md)
- [对比学习Mask增强](docs/train_thermal_classifier4_README.md)
- [分类方法对比报告](docs/thermal_classification_methods_report.md)

---
*文档生成时间: 2025-10-11*
*重命名脚本版本: v1.0*
"""
    
    summary_path = Path('model/docs/thermal_classifiers_rename_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"📄 功能总结文档已创建: {summary_path}")

def main():
    """主函数"""
    print("🚀 热力图分类器重命名工具")
    print("=" * 60)
    
    # 确认操作
    response = input("确认要重命名所有分类器脚本吗? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # 执行重命名
        rename_thermal_classifiers()
        
        # 创建总结文档
        create_classifier_summary()
        
        print("\n🎉 重命名完成!")
        print("💡 建议:")
        print("  1. 检查并更新任何引用旧脚本名的代码")
        print("  2. 更新相关文档和配置文件")
        print("  3. 查看生成的功能总结文档")
        
    else:
        print("❌ 操作已取消")

if __name__ == "__main__":
    main()
