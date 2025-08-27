#!/usr/bin/env python3
"""
全面的特征重要性统计分析脚本
分析所有图像的Grad-CAM结果，统计特征层重要性和位置模式
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from run_interpretability_analysis import find_latest_model, find_dataset_dir, detect_model_asymmetry_mode
from interpretability_analysis import ThermalInterpretabilityAnalyzer

class FeatureImportanceAnalyzer:
    """特征重要性统计分析器"""
    
    def __init__(self, output_dir: str = "dataset/datasets/feature_importance_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "heatmap_clusters").mkdir(exist_ok=True)
        (self.output_dir / "layer_comparisons").mkdir(exist_ok=True)
        
        # 分析结果存储
        self.analysis_results = []
        self.layer_statistics = {}
        self.position_patterns = {}
        self.class_differences = {}
        
    def run_comprehensive_analysis(self, model_path: str = None, dataset_dir: str = None):
        """运行全面分析"""
        print("=== 开始全面特征重要性分析 ===\n")
        
        # 1. 自动查找模型和数据集
        if model_path is None:
            model_path =  "model/contrastive_thermal_classifier_results/run_20250827_180427__last/best_classifier.pth"#find_latest_model()
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
        
        # 4. 收集所有图像
        image_paths = self._collect_all_images(dataset_dir)
        print(f"找到 {len(image_paths)} 张图像进行分析")
        
        # 5. 批量分析所有图像 - 添加限制和进度显示
        print("\n开始批量分析...")

        # 限制分析的图像数量，避免过长时间
        max_images = min(200, len(image_paths))  # 最多分析200张图像
        if len(image_paths) > max_images:
            print(f"图像数量过多({len(image_paths)})，限制为前{max_images}张")
            image_paths = image_paths[:max_images]

        results = []
        import time
        start_time = time.time()

        for i, (image_path, true_label) in enumerate(image_paths):
            # 显示进度
            if i % 10 == 0 or i == len(image_paths) - 1:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1) if i > 0 else 0
                remaining = avg_time * (len(image_paths) - i - 1)
                print(f"进度: {i+1}/{len(image_paths)} ({(i+1)/len(image_paths)*100:.1f}%) - "
                      f"已用时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s")

            try:
                result = analyzer.analyze_single_image(str(image_path))
                result['true_label'] = true_label
                result['image_name'] = Path(image_path).name
                results.append(result)
            except Exception as e:
                print(f"  ❌ 分析失败 {Path(image_path).name}: {e}")
                continue

            # 时间限制：如果单张图像分析时间过长，跳过后续
            if i > 0 and (time.time() - start_time) / (i + 1) > 30:  # 平均每张超过30秒
                print(f"⚠️  分析速度过慢，停止后续分析")
                break
        
        self.analysis_results = results
        print(f"\n成功分析 {len(results)} 张图像")
        
        # 6. 统计分析
        print("\n开始统计分析...")
        self._analyze_layer_importance()
        self._analyze_position_patterns()
        self._analyze_class_differences()
        self._analyze_prediction_accuracy()
        
        # 7. 生成可视化
        print("\n生成可视化图表...")
        self._generate_visualizations()
        
        # 8. 保存结果
        print("\n保存分析结果...")
        self._save_comprehensive_results()
        
        print(f"\n=== 分析完成 ===")
        print(f"结果保存到: {self.output_dir}")
        
    def _collect_all_images(self, dataset_dir: str) -> List[Tuple[str, int]]:
        """收集所有图像及其标签 - 随机采样，不按目录偏向"""
        import random

        image_paths = []
        dataset_path = Path(dataset_dir)

        # 收集所有图像文件，不区分目录
        all_image_files = []

        # 递归查找所有jpg文件
        for img_path in dataset_path.glob("**/*.jpg"):
            all_image_files.append(img_path)

        print(f"找到总计 {len(all_image_files)} 张图像")

        # 随机打乱顺序，避免目录偏向
        random.shuffle(all_image_files)

        # 为每张图像分配标签
        for img_path in all_image_files:
            # 尝试从路径推断标签
            path_str = str(img_path).lower()
            if "icas" in path_str and "non_icas" not in path_str:
                label = 1  # ICAS
            elif "non_icas" in path_str or "non-icas" in path_str:
                label = 0  # Non-ICAS
            else:
                # 无法从路径判断，设为未知
                label = -1  # 未知标签

            image_paths.append((str(img_path), label))

        # 统计标签分布
        icas_count = sum(1 for _, label in image_paths if label == 1)
        non_icas_count = sum(1 for _, label in image_paths if label == 0)
        unknown_count = sum(1 for _, label in image_paths if label == -1)

        print(f"标签分布: ICAS={icas_count}, Non-ICAS={non_icas_count}, 未知={unknown_count}")

        return image_paths
    
    def _analyze_layer_importance(self):
        """分析各层重要性"""
        print("  📊 分析层重要性...")
        
        layer_stats = {}
        
        for result in self.analysis_results:
            for layer_name, layer_result in result['gradcam_results'].items():
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        'max_activations': [],
                        'mean_activations': [],
                        'activation_areas': [],
                        'prediction_correct': [],
                        'confidence_scores': []
                    }
                
                stats_data = layer_result['feature_statistics']
                layer_stats[layer_name]['max_activations'].append(stats_data['max_activation'])
                layer_stats[layer_name]['mean_activations'].append(stats_data['mean_activation'])
                layer_stats[layer_name]['activation_areas'].append(stats_data['activation_area'])
                
                # 预测准确性
                pred_correct = (result['predicted_class'] == result['true_label']) if result['true_label'] != -1 else None
                layer_stats[layer_name]['prediction_correct'].append(pred_correct)
                layer_stats[layer_name]['confidence_scores'].append(result['confidence'])
        
        # 计算统计指标
        for layer_name, stats in layer_stats.items():
            layer_summary = {
                'layer_name': layer_name,
                'avg_max_activation': np.mean(stats['max_activations']),
                'std_max_activation': np.std(stats['max_activations']),
                'avg_mean_activation': np.mean(stats['mean_activations']),
                'avg_activation_area': np.mean(stats['activation_areas']),
                'activation_consistency': 1 - np.std(stats['mean_activations']) / (np.mean(stats['mean_activations']) + 1e-8),
                'sample_count': len(stats['max_activations'])
            }
            
            # 计算与预测准确性的相关性
            valid_predictions = [p for p in stats['prediction_correct'] if p is not None]
            if valid_predictions:
                valid_confidences = [c for c, p in zip(stats['confidence_scores'], stats['prediction_correct']) if p is not None]
                layer_summary['accuracy_correlation'] = np.corrcoef(valid_predictions, valid_confidences)[0, 1] if len(valid_predictions) > 1 else 0
            else:
                layer_summary['accuracy_correlation'] = 0
            
            self.layer_statistics[layer_name] = layer_summary
    
    def _analyze_position_patterns(self):
        """分析位置模式 - 优化版本，减少计算时间"""
        print("  📍 分析位置模式...")

        # 限制分析的样本数量，避免过长的计算时间
        max_samples_per_layer = 500  # 每层最多分析100个样本

        for layer_name in self.layer_statistics.keys():
            centers_of_mass = []

            # 收集位置数据，但限制数量
            sample_count = 0
            for result in self.analysis_results:
                if sample_count >= max_samples_per_layer:
                    break

                if layer_name in result['gradcam_results']:
                    stats_data = result['gradcam_results'][layer_name]['feature_statistics']
                    centers_of_mass.append(stats_data['center_of_mass'])
                    sample_count += 1

            if centers_of_mass:
                centers_array = np.array(centers_of_mass)

                # 计算位置统计
                position_stats = {
                    'mean_center': np.mean(centers_array, axis=0).tolist(),
                    'std_center': np.std(centers_array, axis=0).tolist(),
                    'center_spread': np.mean(np.std(centers_array, axis=0)),
                    'sample_count': len(centers_of_mass),
                    'total_available': len([r for r in self.analysis_results if layer_name in r['gradcam_results']])
                }

                # 简化的聚类分析，只在样本数适中时进行
                if 5 <= len(centers_array) <= 50:  # 只在合理范围内进行聚类
                    try:
                        n_clusters = min(3, len(centers_array) // 3)  # 减少聚类数
                        if n_clusters >= 2:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)  # 减少初始化次数
                            clusters = kmeans.fit_predict(centers_array)
                            position_stats['cluster_centers'] = kmeans.cluster_centers_.tolist()
                            position_stats['cluster_labels'] = clusters.tolist()
                        else:
                            position_stats['cluster_centers'] = []
                            position_stats['cluster_labels'] = []
                    except Exception as e:
                        print(f"    聚类分析失败 {layer_name}: {e}")
                        position_stats['cluster_centers'] = []
                        position_stats['cluster_labels'] = []
                else:
                    # 样本数不合适，跳过聚类
                    position_stats['cluster_centers'] = []
                    position_stats['cluster_labels'] = []
                    if len(centers_array) > 50:
                        print(f"    {layer_name}: 样本数过多({len(centers_array)})，跳过聚类分析")

                self.position_patterns[layer_name] = position_stats
                print(f"    {layer_name}: 分析了 {len(centers_of_mass)} 个样本")
            else:
                print(f"    {layer_name}: 无可用样本")
    
    def _analyze_class_differences(self):
        """分析类别差异"""
        print("  🔍 分析类别差异...")
        
        icas_results = [r for r in self.analysis_results if r['true_label'] == 1]
        non_icas_results = [r for r in self.analysis_results if r['true_label'] == 0]
        
        print(f"    ICAS样本: {len(icas_results)}")
        print(f"    Non-ICAS样本: {len(non_icas_results)}")
        
        for layer_name in self.layer_statistics.keys():
            icas_stats = []
            non_icas_stats = []
            
            for result in icas_results:
                if layer_name in result['gradcam_results']:
                    icas_stats.append(result['gradcam_results'][layer_name]['feature_statistics'])
            
            for result in non_icas_results:
                if layer_name in result['gradcam_results']:
                    non_icas_stats.append(result['gradcam_results'][layer_name]['feature_statistics'])
            
            if icas_stats and non_icas_stats:
                # 计算各指标的类别差异
                icas_max_act = [s['max_activation'] for s in icas_stats]
                non_icas_max_act = [s['max_activation'] for s in non_icas_stats]
                
                icas_mean_act = [s['mean_activation'] for s in icas_stats]
                non_icas_mean_act = [s['mean_activation'] for s in non_icas_stats]
                
                icas_area = [s['activation_area'] for s in icas_stats]
                non_icas_area = [s['activation_area'] for s in non_icas_stats]
                
                # 统计检验
                try:
                    max_act_pvalue = stats.ttest_ind(icas_max_act, non_icas_max_act)[1]
                    mean_act_pvalue = stats.ttest_ind(icas_mean_act, non_icas_mean_act)[1]
                    area_pvalue = stats.ttest_ind(icas_area, non_icas_area)[1]
                except:
                    max_act_pvalue = mean_act_pvalue = area_pvalue = 1.0
                
                class_diff = {
                    'icas_samples': len(icas_stats),
                    'non_icas_samples': len(non_icas_stats),
                    'max_activation_diff': np.mean(icas_max_act) - np.mean(non_icas_max_act),
                    'mean_activation_diff': np.mean(icas_mean_act) - np.mean(non_icas_mean_act),
                    'activation_area_diff': np.mean(icas_area) - np.mean(non_icas_area),
                    'max_activation_pvalue': max_act_pvalue,
                    'mean_activation_pvalue': mean_act_pvalue,
                    'activation_area_pvalue': area_pvalue,
                    'significant_differences': []
                }
                
                # 标记显著差异
                if max_act_pvalue < 0.05:
                    class_diff['significant_differences'].append('max_activation')
                if mean_act_pvalue < 0.05:
                    class_diff['significant_differences'].append('mean_activation')
                if area_pvalue < 0.05:
                    class_diff['significant_differences'].append('activation_area')
                
                self.class_differences[layer_name] = class_diff
    
    def _analyze_prediction_accuracy(self):
        """分析预测准确性"""
        print("  🎯 分析预测准确性...")
        
        # 整体准确性
        valid_predictions = [r for r in self.analysis_results if r['true_label'] != -1]
        if valid_predictions:
            correct_predictions = sum(1 for r in valid_predictions if r['predicted_class'] == r['true_label'])
            overall_accuracy = correct_predictions / len(valid_predictions)
            
            # 按类别统计
            icas_predictions = [r for r in valid_predictions if r['true_label'] == 1]
            non_icas_predictions = [r for r in valid_predictions if r['true_label'] == 0]
            
            icas_correct = sum(1 for r in icas_predictions if r['predicted_class'] == r['true_label'])
            non_icas_correct = sum(1 for r in non_icas_predictions if r['predicted_class'] == r['true_label'])
            
            self.prediction_accuracy = {
                'overall_accuracy': overall_accuracy,
                'total_samples': len(valid_predictions),
                'correct_predictions': correct_predictions,
                'icas_accuracy': icas_correct / len(icas_predictions) if icas_predictions else 0,
                'non_icas_accuracy': non_icas_correct / len(non_icas_predictions) if non_icas_predictions else 0,
                'icas_samples': len(icas_predictions),
                'non_icas_samples': len(non_icas_predictions)
            }
            
            print(f"    整体准确率: {overall_accuracy:.4f}")
            print(f"    ICAS准确率: {self.prediction_accuracy['icas_accuracy']:.4f}")
            print(f"    Non-ICAS准确率: {self.prediction_accuracy['non_icas_accuracy']:.4f}")
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        print("  📈 生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 层重要性对比
        self._plot_layer_importance()
        
        # 2. 位置模式分析
        self._plot_position_patterns()
        
        # 3. 类别差异分析
        self._plot_class_differences()
        
        # 4. 预测准确性分析
        self._plot_prediction_analysis()
        
        # 5. 综合热力图
        self._plot_comprehensive_heatmap()
    
    def _plot_layer_importance(self):
        """绘制层重要性图表"""
        if not self.layer_statistics:
            return
            
        layers = list(self.layer_statistics.keys())
        metrics = ['avg_max_activation', 'avg_mean_activation', 'avg_activation_area', 'activation_consistency']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('各层特征重要性分析', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [self.layer_statistics[layer][metric] for layer in layers]
            
            bars = ax.bar(range(len(layers)), values, alpha=0.7)
            ax.set_xlabel('网络层')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} 对比')
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
            
            # 标注数值
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{values[j]:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'layer_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_patterns(self):
        """绘制位置模式图表 - 优化版本"""
        if not self.position_patterns:
            print("    跳过位置模式图表：无数据")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('激活位置模式分析', fontsize=16, fontweight='bold')

            layers = list(self.position_patterns.keys())[:4]  # 最多显示4层

            for i, layer in enumerate(layers):
                if i >= 4:
                    break

                ax = axes[i//2, i%2]
                pattern = self.position_patterns[layer]

                # 绘制中心位置分布
                mean_center = pattern['mean_center']
                std_center = pattern['std_center']

                # 创建散点图显示位置分布
                ax.scatter(mean_center[1], mean_center[0], s=200, c='red', marker='x', label='平均中心')

                # 简化椭圆绘制，避免复杂计算
                try:
                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse(xy=(mean_center[1], mean_center[0]),
                                    width=std_center[1]*2, height=std_center[0]*2,
                                    alpha=0.3, facecolor='blue', label='标准差范围')
                    ax.add_patch(ellipse)
                except:
                    # 如果椭圆绘制失败，跳过
                    pass

                ax.set_xlabel('X坐标')
                ax.set_ylabel('Y坐标')
                ax.set_title(f'{layer.split(".")[-1]} 激活中心分布\n(样本数: {pattern["sample_count"]})')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 隐藏空的子图
            for i in range(len(layers), 4):
                axes[i//2, i%2].axis('off')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations' / 'position_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("    ✅ 位置模式图表已生成")

        except Exception as e:
            print(f"    ❌ 位置模式图表生成失败: {e}")
            plt.close()
    
    def _plot_class_differences(self):
        """绘制类别差异图表"""
        if not self.class_differences:
            return
            
        layers = list(self.class_differences.keys())
        metrics = ['max_activation_diff', 'mean_activation_diff', 'activation_area_diff']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(layers))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [self.class_differences[layer][metric] for layer in layers]
            pvalues = [self.class_differences[layer][metric.replace('_diff', '_pvalue')] for layer in layers]
            
            bars = ax.bar(x + i*width, values, width, label=metric.replace('_diff', '').replace('_', ' ').title())
            
            # 标记显著性
            for j, (bar, pval) in enumerate(zip(bars, pvalues)):
                if pval < 0.05:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           '*', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax.set_xlabel('网络层')
        ax.set_ylabel('ICAS - Non-ICAS 差异')
        ax.set_title('各层特征的类别差异分析 (* p<0.05)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'class_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_analysis(self):
        """绘制预测分析图表"""
        if not hasattr(self, 'prediction_accuracy'):
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 整体准确率
        acc_data = self.prediction_accuracy
        categories = ['Overall', 'ICAS', 'Non-ICAS']
        accuracies = [acc_data['overall_accuracy'], acc_data['icas_accuracy'], acc_data['non_icas_accuracy']]
        
        bars1 = ax1.bar(categories, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('准确率')
        ax1.set_title('预测准确率分析')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. 样本分布
        sample_counts = [acc_data['icas_samples'], acc_data['non_icas_samples']]
        labels = ['ICAS', 'Non-ICAS']
        
        ax2.pie(sample_counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('样本分布')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_heatmap(self):
        """绘制综合热力图"""
        if not self.layer_statistics or not self.class_differences:
            return
            
        # 创建综合指标矩阵
        layers = list(self.layer_statistics.keys())
        metrics = ['avg_max_activation', 'avg_mean_activation', 'activation_consistency', 'max_activation_diff']
        
        data_matrix = []
        for layer in layers:
            row = []
            layer_stats = self.layer_statistics[layer]
            row.extend([
                layer_stats['avg_max_activation'],
                layer_stats['avg_mean_activation'], 
                layer_stats['activation_consistency']
            ])
            
            if layer in self.class_differences:
                row.append(abs(self.class_differences[layer]['max_activation_diff']))
            else:
                row.append(0)
            
            data_matrix.append(row)
        
        # 标准化数据
        data_matrix = np.array(data_matrix)
        data_normalized = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8)
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_normalized.T, 
                   xticklabels=[l.split('.')[-1] for l in layers],
                   yticklabels=[m.replace('_', ' ').title() for m in metrics],
                   annot=True, fmt='.3f', cmap='YlOrRd')
        
        plt.title('各层特征重要性综合热力图', fontsize=14, fontweight='bold')
        plt.xlabel('网络层')
        plt.ylabel('特征指标')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_results(self):
        """保存综合分析结果"""
        # 保存详细统计结果
        comprehensive_results = {
            'layer_statistics': self.layer_statistics,
            'position_patterns': self.position_patterns,
            'class_differences': self.class_differences,
            'prediction_accuracy': getattr(self, 'prediction_accuracy', {}),
            'analysis_summary': self._generate_analysis_summary()
        }
        
        with open(self.output_dir / 'comprehensive_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        # 生成文本报告
        self._generate_text_report()
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """生成分析摘要"""
        if not self.layer_statistics:
            return {}
            
        # 找出最重要的层
        layer_importance_scores = {}
        for layer, stats in self.layer_statistics.items():
            # 综合评分：激活强度 + 一致性 + 类别区分度
            score = (stats['avg_max_activation'] * 0.3 + 
                    stats['activation_consistency'] * 0.3)
            
            if layer in self.class_differences:
                class_diff = self.class_differences[layer]
                significant_count = len(class_diff['significant_differences'])
                score += significant_count * 0.4 / 3  # 最多3个显著差异
            
            layer_importance_scores[layer] = score
        
        # 排序找出最重要的层
        sorted_layers = sorted(layer_importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'most_important_layer': sorted_layers[0][0] if sorted_layers else None,
            'layer_importance_ranking': [layer for layer, score in sorted_layers],
            'layer_importance_scores': layer_importance_scores,
            'total_samples_analyzed': len(self.analysis_results),
            'layers_analyzed': len(self.layer_statistics)
        }
    
    def _generate_text_report(self):
        """生成文本分析报告"""
        report_path = self.output_dir / 'comprehensive_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 全面特征重要性分析报告 ===\n\n")
            
            # 基本信息
            f.write(f"分析样本总数: {len(self.analysis_results)}\n")
            f.write(f"分析层数: {len(self.layer_statistics)}\n\n")
            
            # 预测准确性
            if hasattr(self, 'prediction_accuracy'):
                acc = self.prediction_accuracy
                f.write("=== 预测准确性 ===\n")
                f.write(f"整体准确率: {acc['overall_accuracy']:.4f}\n")
                f.write(f"ICAS准确率: {acc['icas_accuracy']:.4f} ({acc['icas_samples']} 样本)\n")
                f.write(f"Non-ICAS准确率: {acc['non_icas_accuracy']:.4f} ({acc['non_icas_samples']} 样本)\n\n")
            
            # 层重要性排名
            if 'analysis_summary' in self.__dict__ or hasattr(self, 'layer_statistics'):
                summary = self._generate_analysis_summary()
                f.write("=== 层重要性排名 ===\n")
                for i, layer in enumerate(summary['layer_importance_ranking'], 1):
                    score = summary['layer_importance_scores'][layer]
                    f.write(f"{i}. {layer}: {score:.4f}\n")
                f.write(f"\n最重要的层: {summary['most_important_layer']}\n\n")
            
            # 显著类别差异
            f.write("=== 显著类别差异 ===\n")
            for layer, diff in self.class_differences.items():
                if diff['significant_differences']:
                    f.write(f"{layer}:\n")
                    for sig_diff in diff['significant_differences']:
                        pvalue = diff[f"{sig_diff}_pvalue"]
                        f.write(f"  - {sig_diff}: p={pvalue:.4f}\n")
            
            f.write(f"\n报告生成时间: {pd.Timestamp.now()}\n")
            f.write(f"结果保存目录: {self.output_dir}\n")

def main():
    """主函数"""
    print("全面特征重要性统计分析工具\n")
    
    analyzer = FeatureImportanceAnalyzer()
    
    try:
        analyzer.run_comprehensive_analysis()
        print("\n🎉 全面分析完成!")
        print(f"📊 查看结果: {analyzer.output_dir}")
        print("📈 可视化图表: visualizations/ 目录")
        print("📋 详细报告: comprehensive_analysis_report.txt")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 使用示例
"""
# 1. 直接运行全面分析（自动查找模型和数据集）
python model/comprehensive_feature_analysis.py

# 2. 指定模型和数据集路径
from model.comprehensive_feature_analysis import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
analyzer.run_comprehensive_analysis(
    model_path="path/to/your/model.pth",
    dataset_dir="path/to/your/dataset"
)
"""
