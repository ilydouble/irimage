import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os
from typing import List, Tuple, Optional
import argparse

# 导入对比学习模型
from train_thermal_classifier3 import ThermalEncoder, ContrastiveThermalDataset

class GradCAM:
    """Grad-CAM可视化类"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 找到目标层
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
        
        # 注册钩子
        self.hooks.append(target_module.register_forward_hook(forward_hook))
        self.hooks.append(target_module.register_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """生成Grad-CAM热力图"""
        self.model.eval()
        
        # 前向传播
        output = self.model.classify(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # 计算Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # 全局平均池化得到权重
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # 加权求和
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化到[0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()

class ThermalInterpretabilityAnalyzer:
    """热力图可解释性分析器"""
    
    def __init__(self, model_path: str, use_asymmetry_analysis: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_asymmetry_analysis = use_asymmetry_analysis
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 设置数据变换
        if use_asymmetry_analysis:
            resize_size = (224, 112)
        else:
            resize_size = (224, 224)
            
        self.transform = T.Compose([
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 创建输出目录
        self.output_dir = Path("dataset/datasets/interpretability_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "gradcam_heatmaps").mkdir(exist_ok=True)
        (self.output_dir / "overlay_images").mkdir(exist_ok=True)
        (self.output_dir / "feature_maps").mkdir(exist_ok=True)
    
    def _load_model(self, model_path: str) -> ThermalEncoder:
        """加载训练好的模型"""
        model = ThermalEncoder(backbone='resnet18').to(self.device)
        
        # 如果使用不对称分析，调整模型结构
        if self.use_asymmetry_analysis:
            original_conv1 = model.backbone.conv1
            model.backbone.conv1 = nn.Conv2d(
                6, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            ).to(self.device)
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def _preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """预处理图像"""
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        original_image = np.array(image)
        
        if self.use_asymmetry_analysis:
            # 分割左右脸
            width, height = image.size
            center_x = width // 2
            
            left_half = image.crop((0, 0, center_x, height))
            right_half = image.crop((center_x, 0, width, height))
            
            # 应用变换
            left_tensor = self.transform(left_half)
            right_tensor = self.transform(right_half)
            
            # 拼接通道
            input_tensor = torch.cat([left_tensor, right_tensor], dim=0)
        else:
            input_tensor = self.transform(image)
        
        # 添加batch维度
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        return input_tensor, original_image
    
    def analyze_single_image(self, image_path: str, target_layers: List[str] = None) -> dict:
        """分析单张图像"""
        if target_layers is None:
            target_layers = [
                'backbone.layer1.1.conv2',  # 浅层特征
                'backbone.layer2.1.conv2',  # 中层特征
                'backbone.layer3.1.conv2',  # 深层特征
                'backbone.layer4.1.conv2'   # 最深层特征
            ]
        
        print(f"分析图像: {image_path}")
        
        # 预处理图像
        input_tensor, original_image = self._preprocess_image(image_path)
        
        # 获取预测结果
        with torch.no_grad():
            prediction = self.model.classify(input_tensor)
            predicted_class = prediction.argmax(dim=1).item()
            confidence = F.softmax(prediction, dim=1).max().item()
        
        print(f"预测类别: {predicted_class} ({'ICAS' if predicted_class == 1 else 'Non-ICAS'})")
        print(f"置信度: {confidence:.4f}")
        
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'gradcam_results': {}
        }
        
        # 对每个目标层生成Grad-CAM
        for layer_name in target_layers:
            print(f"生成 {layer_name} 的Grad-CAM...")
            
            try:
                # 创建Grad-CAM对象
                gradcam = GradCAM(self.model, layer_name)
                
                # 生成热力图
                cam = gradcam.generate_cam(input_tensor, predicted_class)
                
                # 保存结果
                layer_results = self._save_gradcam_results(
                    cam, original_image, image_path, layer_name, predicted_class
                )
                
                results['gradcam_results'][layer_name] = layer_results
                
                # 清理钩子
                gradcam.cleanup()
                
            except Exception as e:
                print(f"处理层 {layer_name} 时出错: {e}")
                continue
        
        return results
    
    def _save_gradcam_results(self, cam: np.ndarray, original_image: np.ndarray, 
                             image_path: str, layer_name: str, predicted_class: int) -> dict:
        """保存Grad-CAM结果"""
        image_name = Path(image_path).stem
        layer_clean = layer_name.replace('.', '_')
        class_name = 'ICAS' if predicted_class == 1 else 'Non_ICAS'
        
        # 1. 保存原始热力图
        heatmap_path = self.output_dir / "gradcam_heatmaps" / f"{image_name}_{layer_clean}_{class_name}_heatmap.png"
        self._save_heatmap(cam, heatmap_path)
        
        # 2. 生成并保存叠加图像
        overlay_path = self.output_dir / "overlay_images" / f"{image_name}_{layer_clean}_{class_name}_overlay.png"
        self._save_overlay_image(cam, original_image, overlay_path)
        
        # 3. 保存特征图统计
        feature_stats = self._compute_feature_statistics(cam)
        
        return {
            'heatmap_path': str(heatmap_path),
            'overlay_path': str(overlay_path),
            'feature_statistics': feature_stats
        }
    
    def _save_heatmap(self, cam: np.ndarray, save_path: Path):
        """保存热力图"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cam, cmap='jet')
        plt.colorbar()
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_overlay_image(self, cam: np.ndarray, original_image: np.ndarray, save_path: Path):
        """保存叠加图像"""
        # 调整热力图尺寸到原图大小
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 将热力图转换为彩色
        heatmap = cm.jet(cam_resized)[:, :, :3]  # 去掉alpha通道
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # 增加原图透明度，减少鲜艳度
        original_alpha = 0.3  # 原图透明度
        heatmap_alpha = 0.7   # 热力图透明度
        
        # 叠加图像
        overlay = cv2.addWeighted(
            original_image.astype(np.uint8), original_alpha,
            heatmap, heatmap_alpha, 0
        )
        
        # 保存
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), overlay_bgr)
        
        # 同时保存matplotlib版本（更好的颜色映射）
        plt.figure(figsize=(10, 8))
        plt.imshow(original_image, alpha=original_alpha)
        plt.imshow(cam_resized, cmap='jet', alpha=heatmap_alpha)
        plt.axis('off')
        plt.title('Grad-CAM Overlay')
        
        # 保存matplotlib版本
        plt_save_path = save_path.parent / f"{save_path.stem}_plt.png"
        plt.savefig(plt_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def _compute_feature_statistics(self, cam: np.ndarray) -> dict:
        """计算特征统计信息"""
        return {
            'max_activation': float(cam.max()),
            'mean_activation': float(cam.mean()),
            'std_activation': float(cam.std()),
            'activation_area': float((cam > 0.5).sum() / cam.size),  # 高激活区域比例
            'center_of_mass': [float(x) for x in np.unravel_index(cam.argmax(), cam.shape)]
        }
    
    def batch_analyze(self, image_dir: str, pattern: str = "*.jpg") -> List[dict]:
        """批量分析图像"""
        image_paths = list(Path(image_dir).glob(pattern))
        print(f"找到 {len(image_paths)} 张图像进行分析")
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"\n进度: {i+1}/{len(image_paths)}")
            try:
                result = self.analyze_single_image(str(image_path))
                results.append(result)
            except Exception as e:
                print(f"分析图像 {image_path} 时出错: {e}")
                continue
        
        # 保存批量分析结果
        self._save_batch_results(results)
        
        return results
    
    def _save_batch_results(self, results: List[dict]):
        """保存批量分析结果"""
        import json
        
        # 保存详细结果
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成汇总报告
        self._generate_summary_report(results)
    
    def _generate_summary_report(self, results: List[dict]):
        """生成汇总报告"""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 可解释性分析汇总报告 ===\n\n")
            f.write(f"分析图像总数: {len(results)}\n")
            
            # 预测统计
            icas_count = sum(1 for r in results if r['predicted_class'] == 1)
            non_icas_count = len(results) - icas_count
            f.write(f"预测为ICAS: {icas_count} ({icas_count/len(results)*100:.1f}%)\n")
            f.write(f"预测为Non-ICAS: {non_icas_count} ({non_icas_count/len(results)*100:.1f}%)\n")
            
            # 置信度统计
            confidences = [r['confidence'] for r in results]
            f.write(f"平均置信度: {np.mean(confidences):.4f}\n")
            f.write(f"置信度标准差: {np.std(confidences):.4f}\n")
            
            f.write(f"\n输出目录: {self.output_dir}\n")
            f.write("- gradcam_heatmaps/: Grad-CAM热力图\n")
            f.write("- overlay_images/: 叠加可视化图像\n")
            f.write("- feature_maps/: 特征图统计\n")

def main():
    parser = argparse.ArgumentParser(description="热力图对比学习模型可解释性分析")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--image_path", type=str, help="单张图像路径")
    parser.add_argument("--image_dir", type=str, help="图像目录路径")
    parser.add_argument("--use_asymmetry", action="store_true", default=True, help="使用不对称分析")
    parser.add_argument("--pattern", type=str, default="*.jpg", help="图像文件模式")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ThermalInterpretabilityAnalyzer(
        model_path=args.model_path,
        use_asymmetry_analysis=args.use_asymmetry
    )
    
    if args.image_path:
        # 分析单张图像
        result = analyzer.analyze_single_image(args.image_path)
        print(f"\n分析完成，结果保存到: {analyzer.output_dir}")
        
    elif args.image_dir:
        # 批量分析
        results = analyzer.batch_analyze(args.image_dir, args.pattern)
        print(f"\n批量分析完成，共处理 {len(results)} 张图像")
        print(f"结果保存到: {analyzer.output_dir}")
        
    else:
        print("请指定 --image_path 或 --image_dir")

if __name__ == "__main__":
    main()

# 使用示例
"""
# 1. 分析单张图像
python model/interpretability_analysis.py \
    --model_path "./model/contrastive_thermal_classifier_results/run_20250826_234950/best_contrastive_encoder.pth" \
    --image_path "./dataset/datasets/thermal_classification_cropped/icas/patient_001.jpg" \
    --use_asymmetry

# 2. 批量分析目录中的图像
python model/interpretability_analysis.py \
    --model_path "./model/contrastive_thermal_classifier_results/run_20250826_234950/best_contrastive_encoder.pth" \
    --image_dir "./dataset/datasets/thermal_classification_cropped/icas/" \
    --pattern "*.jpg" \
    --use_asymmetry

# 3. 分析所有类别
python model/interpretability_analysis.py \
    --model_path "./model/contrastive_thermal_classifier_results/run_20250826_234950/best_contrastive_encoder.pth" \
    --image_dir "./dataset/datasets/thermal_classification_cropped/" \
    --pattern "*/*.jpg" \
    --use_asymmetry
"""
