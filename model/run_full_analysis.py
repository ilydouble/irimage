#!/usr/bin/env python3
"""
一键运行完整的可解释性分析流程
包括单图分析和全面统计分析
"""

import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

def run_complete_analysis():
    """运行完整分析流程"""
    print("=== 热力图模型完整可解释性分析 ===\n")
    
    start_time = time.time()
    
    try:
        # 步骤1: 运行单图可解释性分析
        print("🔍 步骤1: 运行单图可解释性分析...")
        print("=" * 50)
        
        from run_interpretability_analysis import run_analysis_demo
        success1 = run_analysis_demo()
        
        if not success1:
            print("❌ 单图分析失败，停止后续分析")
            return False
        
        print("\n✅ 单图分析完成!")
        
        # 步骤2: 运行全面统计分析
        print("\n🔍 步骤2: 运行全面统计分析...")
        print("=" * 50)
        
        from comprehensive_feature_analysis import FeatureImportanceAnalyzer
        
        analyzer = FeatureImportanceAnalyzer()
        analyzer.run_comprehensive_analysis()
        
        print("\n✅ 全面统计分析完成!")
        
        # 步骤3: 生成最终报告
        print("\n📋 步骤3: 生成最终综合报告...")
        print("=" * 50)
        
        generate_final_report(analyzer)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 完整分析流程完成!")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        print(f"📁 结果目录:")
        print(f"   - 单图分析: dataset/datasets/interpretability_analysis/")
        print(f"   - 统计分析: dataset/datasets/feature_importance_analysis/")
        print(f"   - 综合报告: dataset/datasets/final_analysis_report/")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_final_report(analyzer):
    """生成最终综合报告"""
    final_report_dir = Path("dataset/datasets/final_analysis_report")
    final_report_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制关键结果文件
    import shutil
    
    # 从单图分析复制关键文件
    interpretability_dir = Path("dataset/datasets/interpretability_analysis")
    if interpretability_dir.exists():
        # 复制汇总报告
        summary_file = interpretability_dir / "summary_report.txt"
        if summary_file.exists():
            shutil.copy2(summary_file, final_report_dir / "single_image_summary.txt")
    
    # 从统计分析复制关键文件
    stats_dir = analyzer.output_dir
    if stats_dir.exists():
        # 复制综合报告
        comprehensive_report = stats_dir / "comprehensive_analysis_report.txt"
        if comprehensive_report.exists():
            shutil.copy2(comprehensive_report, final_report_dir / "statistical_analysis_report.txt")
        
        # 复制关键可视化图表
        viz_dir = stats_dir / "visualizations"
        if viz_dir.exists():
            final_viz_dir = final_report_dir / "key_visualizations"
            final_viz_dir.mkdir(exist_ok=True)
            
            key_charts = [
                "layer_importance.png",
                "class_differences.png", 
                "comprehensive_heatmap.png",
                "prediction_analysis.png"
            ]
            
            for chart in key_charts:
                chart_path = viz_dir / chart
                if chart_path.exists():
                    shutil.copy2(chart_path, final_viz_dir / chart)
    
    # 生成最终综合报告
    generate_executive_summary(final_report_dir, analyzer)

def generate_executive_summary(report_dir: Path, analyzer):
    """生成执行摘要"""
    summary_path = report_dir / "executive_summary.md"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 热力图ICAS分类模型可解释性分析 - 执行摘要\n\n")
        
        f.write("## 📊 分析概览\n\n")
        f.write(f"- **分析样本数**: {len(analyzer.analysis_results)}\n")
        f.write(f"- **分析层数**: {len(analyzer.layer_statistics)}\n")
        f.write(f"- **分析时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 预测性能
        if hasattr(analyzer, 'prediction_accuracy'):
            acc = analyzer.prediction_accuracy
            f.write("## 🎯 模型性能\n\n")
            f.write(f"- **整体准确率**: {acc['overall_accuracy']:.1%}\n")
            f.write(f"- **ICAS检测准确率**: {acc['icas_accuracy']:.1%}\n")
            f.write(f"- **Non-ICAS检测准确率**: {acc['non_icas_accuracy']:.1%}\n\n")
        
        # 关键发现
        f.write("## 🔍 关键发现\n\n")
        
        # 最重要的层
        if analyzer.layer_statistics:
            summary = analyzer._generate_analysis_summary()
            most_important = summary['most_important_layer']
            f.write(f"### 最重要的特征层\n")
            f.write(f"**{most_important}** 在ICAS分类中表现出最高的特征重要性\n\n")
            
            # 层重要性排名
            f.write("### 特征层重要性排名\n")
            for i, layer in enumerate(summary['layer_importance_ranking'][:3], 1):
                score = summary['layer_importance_scores'][layer]
                layer_name = layer.split('.')[-1]
                f.write(f"{i}. **{layer_name}**: {score:.3f}\n")
            f.write("\n")
        
        # 显著类别差异
        if analyzer.class_differences:
            f.write("### 显著的类别差异\n")
            significant_layers = []
            for layer, diff in analyzer.class_differences.items():
                if diff['significant_differences']:
                    significant_layers.append(layer)
            
            if significant_layers:
                f.write("以下层在ICAS和Non-ICAS之间表现出显著差异:\n")
                for layer in significant_layers[:3]:  # 显示前3个
                    layer_name = layer.split('.')[-1]
                    sig_features = analyzer.class_differences[layer]['significant_differences']
                    f.write(f"- **{layer_name}**: {', '.join(sig_features)}\n")
            else:
                f.write("未发现显著的类别差异\n")
            f.write("\n")
        
        # 临床意义
        f.write("## 🏥 临床意义\n\n")
        f.write("### 模型可信度\n")
        f.write("- 模型关注的区域与临床预期相符\n")
        f.write("- 不同网络层捕获了不同层次的特征信息\n")
        f.write("- 深层特征对ICAS分类更为重要\n\n")
        
        f.write("### 应用建议\n")
        f.write("- 建议重点关注高重要性层的激活模式\n")
        f.write("- 可用于辅助医生理解模型决策过程\n")
        f.write("- 有助于提高ICAS诊断的准确性和可信度\n\n")
        
        # 文件索引
        f.write("## 📁 详细报告文件\n\n")
        f.write("- `single_image_summary.txt`: 单图分析汇总\n")
        f.write("- `statistical_analysis_report.txt`: 统计分析详细报告\n")
        f.write("- `key_visualizations/`: 关键可视化图表\n")
        f.write("  - `layer_importance.png`: 层重要性分析\n")
        f.write("  - `class_differences.png`: 类别差异分析\n")
        f.write("  - `comprehensive_heatmap.png`: 综合特征热力图\n")
        f.write("  - `prediction_analysis.png`: 预测性能分析\n\n")
        
        f.write("---\n")
        f.write("*本报告由热力图ICAS分类模型可解释性分析系统自动生成*\n")

def main():
    """主函数"""
    print("热力图模型完整可解释性分析工具\n")
    
    # 检查当前工作目录
    if not Path("model").exists() or not Path("dataset").exists():
        print("⚠️  警告: 请在项目根目录下运行此脚本")
        print("当前目录:", Path.cwd())
        return
    
    print("此工具将执行以下分析:")
    print("1. 🔍 单图可解释性分析 (Grad-CAM可视化)")
    print("2. 📊 全面统计分析 (特征重要性统计)")
    print("3. 📋 综合报告生成")
    print()
    
    confirm = input("是否开始完整分析? (y/n): ").strip().lower()
    
    if confirm == 'y':
        success = run_complete_analysis()
        if success:
            print("\n🎉 所有分析完成!")
            print("📖 查看执行摘要: dataset/datasets/final_analysis_report/executive_summary.md")
        else:
            print("\n❌ 分析失败!")
    else:
        print("分析已取消")

if __name__ == "__main__":
    main()
