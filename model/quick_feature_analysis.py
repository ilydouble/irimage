#!/usr/bin/env python3
"""
快速特征重要性分析脚本
专门用于快速测试，避免长时间等待
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from run_interpretability_analysis import find_latest_model, find_dataset_dir, detect_model_asymmetry_mode
from interpretability_analysis import ThermalInterpretabilityAnalyzer

class QuickFeatureAnalyzer:
    """快速特征重要性分析器"""
    
    def __init__(self, output_dir: str = "dataset/datasets/quick_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analysis_results = []
        self.layer_statistics = {}
        
    def run_quick_analysis(self, model_path: str = None, dataset_dir: str = None, max_images: int = 20):
        """运行快速分析，限制图像数量"""
        print("=== 快速特征重要性分析 ===\n")
        
        start_time = time.time()
        
        # 1. 自动查找模型和数据集
        if model_path is None:
            model_path = "model/contrastive_thermal_classifier_results/run_20250827_180427__last/best_classifier.pth"
        if dataset_dir is None:
            dataset_dir = find_dataset_dir()
            
        print(f"模型路径: {model_path}")
        print(f"数据集路径: {dataset_dir}")
        
        # 2. 检测模型模式
        use_asymmetry = detect_model_asymmetry_mode(model_path)
        print(f"模型模式: {'不对称分析 (6通道)' if use_asymmetry else '标准模式 (3通道)'}")
        
        # 3. 创建分析器
        analyzer = ThermalInterpretabilityAnalyzer(
            model_path=model_path,
            use_asymmetry_analysis=use_asymmetry
        )
        
        # 4. 收集少量图像进行快速测试
        image_paths = self._collect_sample_images(dataset_dir, max_images)
        print(f"选择 {len(image_paths)} 张图像进行快速分析")
        
        # 5. 快速分析
        print("\n开始快速分析...")
        results = []
        for i, (image_path, true_label) in enumerate(image_paths):
            print(f"进度: {i+1}/{len(image_paths)} - {Path(image_path).name}")
            try:
                result = analyzer.analyze_single_image(str(image_path))
                result['true_label'] = true_label
                result['image_name'] = Path(image_path).name
                results.append(result)
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                continue
        
        self.analysis_results = results
        print(f"\n成功分析 {len(results)} 张图像")
        
        # 6. 快速统计分析
        print("\n开始统计分析...")
        self._quick_layer_analysis()
        
        # 7. 生成简化报告
        print("\n生成分析报告...")
        self._generate_quick_report()
        
        total_time = time.time() - start_time
        print(f"\n=== 快速分析完成 ===")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"结果保存到: {self.output_dir}")
        
    def _collect_sample_images(self, dataset_dir: str, max_images: int) -> List[Tuple[str, int]]:
        """收集样本图像 - 随机采样，不按目录偏向"""
        import random

        image_paths = []
        dataset_path = Path(dataset_dir)

        # 收集所有图像文件，不区分目录
        all_image_files = list(dataset_path.glob("**/*.jpg"))
        print(f"找到总计 {len(all_image_files)} 张图像")

        # 随机打乱顺序，避免目录偏向
        random.shuffle(all_image_files)

        # 限制数量
        selected_images = all_image_files[:max_images]

        # 为每张图像分配标签
        icas_count = 0
        non_icas_count = 0
        unknown_count = 0

        for img_path in selected_images:
            # 尝试从路径推断标签
            path_str = str(img_path).lower()
            if "icas" in path_str and "non_icas" not in path_str:
                label = 1  # ICAS
                icas_count += 1
            elif "non_icas" in path_str or "non-icas" in path_str:
                label = 0  # Non-ICAS
                non_icas_count += 1
            else:
                # 无法从路径判断，设为未知
                label = -1  # 未知标签
                unknown_count += 1

            image_paths.append((str(img_path), label))

        print(f"  随机选择的样本分布:")
        print(f"  ICAS: {icas_count} 张")
        print(f"  Non-ICAS: {non_icas_count} 张")
        print(f"  未知: {unknown_count} 张")

        return image_paths
    
    def _quick_layer_analysis(self):
        """快速层重要性分析"""
        print("  📊 分析层重要性...")
        
        layer_stats = {}
        
        for result in self.analysis_results:
            for layer_name, layer_result in result['gradcam_results'].items():
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        'max_activations': [],
                        'mean_activations': [],
                        'activation_areas': [],
                        'confidence_scores': []
                    }
                
                stats_data = layer_result['feature_statistics']
                layer_stats[layer_name]['max_activations'].append(stats_data['max_activation'])
                layer_stats[layer_name]['mean_activations'].append(stats_data['mean_activation'])
                layer_stats[layer_name]['activation_areas'].append(stats_data['activation_area'])
                layer_stats[layer_name]['confidence_scores'].append(result['confidence'])
        
        # 计算统计指标
        for layer_name, stats in layer_stats.items():
            layer_summary = {
                'layer_name': layer_name,
                'avg_max_activation': np.mean(stats['max_activations']),
                'avg_mean_activation': np.mean(stats['mean_activations']),
                'avg_activation_area': np.mean(stats['activation_areas']),
                'activation_consistency': 1 - np.std(stats['mean_activations']) / (np.mean(stats['mean_activations']) + 1e-8),
                'avg_confidence': np.mean(stats['confidence_scores']),
                'sample_count': len(stats['max_activations'])
            }
            
            self.layer_statistics[layer_name] = layer_summary
    
    def _generate_quick_report(self):
        """生成快速分析报告"""
        # 1. 保存详细结果
        results_data = {
            'analysis_results': self.analysis_results,
            'layer_statistics': self.layer_statistics,
            'analysis_summary': self._generate_summary()
        }
        
        with open(self.output_dir / 'quick_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 2. 生成文本报告
        report_path = self.output_dir / 'quick_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 快速特征重要性分析报告 ===\n\n")
            
            # 基本信息
            f.write(f"分析样本数: {len(self.analysis_results)}\n")
            f.write(f"分析层数: {len(self.layer_statistics)}\n\n")
            
            # 层重要性排名
            if self.layer_statistics:
                # 按平均最大激活排序
                sorted_layers = sorted(
                    self.layer_statistics.items(), 
                    key=lambda x: x[1]['avg_max_activation'], 
                    reverse=True
                )
                
                f.write("=== 层重要性排名 (按平均最大激活) ===\n")
                for i, (layer, stats) in enumerate(sorted_layers, 1):
                    f.write(f"{i}. {layer}: {stats['avg_max_activation']:.4f}\n")
                f.write("\n")
                
                # 详细统计
                f.write("=== 详细层统计 ===\n")
                for layer, stats in sorted_layers:
                    f.write(f"\n{layer}:\n")
                    f.write(f"  平均最大激活: {stats['avg_max_activation']:.4f}\n")
                    f.write(f"  平均激活值: {stats['avg_mean_activation']:.4f}\n")
                    f.write(f"  平均激活区域: {stats['avg_activation_area']:.4f}\n")
                    f.write(f"  激活一致性: {stats['activation_consistency']:.4f}\n")
                    f.write(f"  平均置信度: {stats['avg_confidence']:.4f}\n")
                    f.write(f"  样本数: {stats['sample_count']}\n")
            
            f.write(f"\n报告生成时间: {pd.Timestamp.now()}\n")
        
        # 3. 生成简单可视化
        self._generate_quick_visualization()
    
    def _generate_quick_visualization(self):
        """生成快速可视化"""
        if not self.layer_statistics:
            return
        
        try:
            layers = list(self.layer_statistics.keys())
            max_activations = [self.layer_statistics[layer]['avg_max_activation'] for layer in layers]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(layers)), max_activations, alpha=0.7)
            plt.xlabel('网络层')
            plt.ylabel('平均最大激活值')
            plt.title('各层特征重要性快速分析')
            plt.xticks(range(len(layers)), [l.split('.')[-1] for l in layers], rotation=45)
            
            # 标注数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{max_activations[i]:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'quick_layer_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("  ✅ 快速可视化已生成")
            
        except Exception as e:
            print(f"  ❌ 可视化生成失败: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成分析摘要"""
        if not self.layer_statistics:
            return {}
        
        # 找出最重要的层
        best_layer = max(
            self.layer_statistics.items(), 
            key=lambda x: x[1]['avg_max_activation']
        )
        
        return {
            'most_important_layer': best_layer[0],
            'highest_activation': best_layer[1]['avg_max_activation'],
            'total_samples': len(self.analysis_results),
            'layers_analyzed': len(self.layer_statistics)
        }

def main():
    """主函数"""
    print("快速特征重要性分析工具\n")
    
    # 获取用户输入
    try:
        max_images = int(input("请输入要分析的图像数量 (默认20): ") or "20")
    except:
        max_images = 20
    
    analyzer = QuickFeatureAnalyzer()
    
    try:
        analyzer.run_quick_analysis(max_images=max_images)
        print("\n🎉 快速分析完成!")
        print(f"📊 查看结果: {analyzer.output_dir}")
        print("📋 详细报告: quick_analysis_report.txt")
        print("📈 可视化图表: quick_layer_importance.png")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
