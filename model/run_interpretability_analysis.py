#!/usr/bin/env python3
"""
热力图对比学习模型可解释性分析运行脚本
简化版本，自动查找最新的模型和数据集
"""

import os
import sys
from pathlib import Path
import glob
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from interpretability_analysis import ThermalInterpretabilityAnalyzer
import torch

def detect_model_asymmetry_mode(model_path: str) -> bool:
    """检测模型是否使用不对称分析模式"""
    try:
        # 加载模型状态字典
        state_dict = torch.load(model_path, map_location='cpu')

        # 检查conv1层的输入通道数
        conv1_weight_key = 'backbone.conv1.weight'
        if conv1_weight_key in state_dict:
            conv1_shape = state_dict[conv1_weight_key].shape
            input_channels = conv1_shape[1]  # [out_channels, in_channels, H, W]

            if input_channels == 6:
                return True  # 不对称分析模式
            elif input_channels == 3:
                return False  # 标准模式
            else:
                print(f"警告: 检测到异常的输入通道数: {input_channels}")
                return False
        else:
            print(f"警告: 在模型中未找到 {conv1_weight_key}")
            return False

    except Exception as e:
        print(f"警告: 检测模型模式时出错: {e}")
        print("默认使用标准模式 (3通道)")
        return False

def find_latest_model(base_dir: str = "./model/contrastive_thermal_classifier_results") -> str:
    """查找最新的训练模型"""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {base_dir}")
    
    # 查找所有运行目录
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        raise FileNotFoundError(f"在 {base_dir} 中未找到运行目录")
    
    # 按时间排序，获取最新的
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    latest_run = run_dirs[0]
    
    # 查找模型文件
    model_files = [
    #    latest_run / "best_contrastive_encoder.pth",
    #    latest_run / "contrastive_encoder.pth",
        latest_run / "best_classifier.pth"
    ]
    
    for model_file in model_files:
        if model_file.exists():
            print(f"找到模型: {model_file}")
            return str(model_file)
    
    raise FileNotFoundError(f"在 {latest_run} 中未找到模型文件")

def find_dataset_dir(base_dir: str = "./dataset/datasets") -> str:
    """查找数据集目录"""
    possible_dirs = [
        Path(base_dir) / "thermal_classification_cropped"
    ]
    
    for dataset_dir in possible_dirs:
        if dataset_dir.exists():
            print(f"找到数据集: {dataset_dir}")
            return str(dataset_dir)
    
    raise FileNotFoundError(f"在 {base_dir} 中未找到数据集目录")

def run_analysis_demo():
    """运行演示分析"""
    print("=== 热力图对比学习模型可解释性分析 ===\n")
    
    try:
        # 1. 查找模型和数据集
        print("1. 查找最新模型和数据集...")
        model_path = find_latest_model()
        dataset_dir = find_dataset_dir()
        
        # 2. 检测模型类型并创建分析器
        print("\n2. 检测模型类型并初始化分析器...")

        # 尝试检测模型是否使用不对称分析
        use_asymmetry = detect_model_asymmetry_mode(model_path)
        print(f"检测到模型模式: {'不对称分析 (6通道)' if use_asymmetry else '标准模式 (3通道)'}")

        analyzer = ThermalInterpretabilityAnalyzer(
            model_path=model_path,
            use_asymmetry_analysis=use_asymmetry
        )
        
        # 3. 查找测试图像
        print("\n3. 查找测试图像...")
        
        # 优先查找ICAS类别的图像
        icas_dir = Path(dataset_dir) / "icas"
        non_icas_dir = Path(dataset_dir) / "non_icas"
        
        test_images = []
        
        # 从ICAS类别选择几张图像
        if icas_dir.exists():
            icas_images = list(icas_dir.glob("*.jpg"))[:3]  # 最多3张
            test_images.extend(icas_images)
            print(f"找到 {len(icas_images)} 张ICAS图像")
        
        # 从Non-ICAS类别选择几张图像
        if non_icas_dir.exists():
            non_icas_images = list(non_icas_dir.glob("*.jpg"))[:3]  # 最多3张
            test_images.extend(non_icas_images)
            print(f"找到 {len(non_icas_images)} 张Non-ICAS图像")
        
        if not test_images:
            # 如果没有分类目录，直接从根目录查找
            test_images = list(Path(dataset_dir).glob("**/*.jpg"))[:6]
            print(f"从根目录找到 {len(test_images)} 张图像")
        
        if not test_images:
            raise FileNotFoundError("未找到测试图像")
        
        print(f"总共将分析 {len(test_images)} 张图像")
        
        # 4. 运行分析
        print("\n4. 开始可解释性分析...")
        results = []
        
        for i, image_path in enumerate(test_images):
            print(f"\n--- 分析图像 {i+1}/{len(test_images)}: {image_path.name} ---")
            try:
                result = analyzer.analyze_single_image(str(image_path))
                results.append(result)
                print(f"✓ 分析完成")
            except Exception as e:
                print(f"✗ 分析失败: {e}")
                continue
        
        # 5. 生成汇总
        print(f"\n5. 生成分析报告...")
        analyzer._save_batch_results(results)
        
        # 6. 显示结果
        print(f"\n=== 分析完成 ===")
        print(f"成功分析: {len(results)} 张图像")
        print(f"输出目录: {analyzer.output_dir}")
        print(f"\n输出文件:")
        print(f"  📊 分析结果: {analyzer.output_dir}/analysis_results.json")
        print(f"  📋 汇总报告: {analyzer.output_dir}/summary_report.txt")
        print(f"  🔥 热力图: {analyzer.output_dir}/gradcam_heatmaps/")
        print(f"  🖼️  叠加图像: {analyzer.output_dir}/overlay_images/")
        
        # 显示一些统计信息
        if results:
            icas_predictions = sum(1 for r in results if r['predicted_class'] == 1)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            print(f"\n📈 预测统计:")
            print(f"  ICAS预测: {icas_predictions}/{len(results)} ({icas_predictions/len(results)*100:.1f}%)")
            print(f"  平均置信度: {avg_confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        return False

def run_custom_analysis():
    """运行自定义分析"""
    print("=== 自定义分析模式 ===\n")
    
    # 获取用户输入
    model_path = input("请输入模型路径 (回车使用自动查找): ").strip()
    if not model_path:
        model_path = find_latest_model()
    
    image_input = input("请输入图像路径或目录 (回车使用默认数据集): ").strip()
    if not image_input:
        image_input = find_dataset_dir()
    
    use_asymmetry_input = input("是否使用不对称分析? (y/n/auto, 默认auto): ").strip().lower()

    if use_asymmetry_input == 'auto' or use_asymmetry_input == '':
        # 自动检测模型模式
        use_asymmetry = detect_model_asymmetry_mode(model_path)
        print(f"自动检测模式: {'不对称分析 (6通道)' if use_asymmetry else '标准模式 (3通道)'}")
    else:
        use_asymmetry = use_asymmetry_input == 'y'

    # 创建分析器
    analyzer = ThermalInterpretabilityAnalyzer(
        model_path=model_path,
        use_asymmetry_analysis=use_asymmetry
    )
    
    # 判断输入是文件还是目录
    input_path = Path(image_input)
    
    if input_path.is_file():
        # 单文件分析
        print(f"分析单张图像: {input_path}")
        _ = analyzer.analyze_single_image(str(input_path))
        print(f"分析完成，结果保存到: {analyzer.output_dir}")
        
    elif input_path.is_dir():
        # 目录批量分析
        pattern = input("请输入文件模式 (默认*.jpg): ").strip()
        if not pattern:
            pattern = "*.jpg"
        
        print(f"批量分析目录: {input_path}")
        print(f"文件模式: {pattern}")
        
        results = analyzer.batch_analyze(str(input_path), pattern)
        print(f"批量分析完成，共处理 {len(results)} 张图像")
        print(f"结果保存到: {analyzer.output_dir}")
        
    else:
        print(f"错误: 路径不存在 - {input_path}")

def main():
    """主函数"""
    print("热力图对比学习模型可解释性分析工具\n")
    
    # 检查当前工作目录
    if not Path("model").exists() or not Path("dataset").exists():
        print("⚠️  警告: 请在项目根目录下运行此脚本")
        print("当前目录:", os.getcwd())
        return
    
    print("请选择运行模式:")
    print("1. 演示模式 (自动查找模型和数据集)")
    print("2. 自定义模式 (手动指定路径)")
    print("3. 退出")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == '1':
        success = run_analysis_demo()
        if success:
            print("\n🎉 演示分析完成!")
        else:
            print("\n❌ 演示分析失败!")
            
    elif choice == '2':
        run_custom_analysis()
        print("\n🎉 自定义分析完成!")
        
    elif choice == '3':
        print("退出程序")
        
    else:
        print("无效选择，退出程序")

if __name__ == "__main__":
    main()
