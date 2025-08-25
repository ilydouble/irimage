import os
import torch
from ultralytics import YOLO
from pathlib import Path
import shutil

class YOLO11Predictor:
    def __init__(self, 
                 model_path="thermal_segmentation/yolo11_thermal_macos/weights/best.pt",
                 source_dir="./dataset/datasets/thermal_25h/non_icas",
                 output_dir="./dataset/datasets/thermal_25h/non_icas_result"):
        
        self.model_path = model_path
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 检查模型文件是否存在
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 检查源目录是否存在
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # macOS设备检测
        self.device = self._get_best_device()
        print(f"使用设备: {self.device}")
        
    def _get_best_device(self):
        """为macOS选择最佳设备"""
        if torch.backends.mps.is_available():
            print("检测到MPS支持，使用Metal GPU加速")
            return 'mps'
        elif torch.cuda.is_available():
            print("检测到CUDA支持")
            return 'cuda'
        else:
            print("使用CPU预测")
            return 'cpu'
    
    def load_model(self):
        """加载训练好的YOLO11模型"""
        print(f"加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("模型加载成功!")
        
    def find_images(self):
        """查找源目录中的所有图像文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        
        for ext in image_extensions:
            images.extend(list(self.source_dir.glob(f'*{ext}')))
            images.extend(list(self.source_dir.glob(f'*{ext.upper()}')))
        
        print(f"找到 {len(images)} 张图像")
        return sorted(images)
    
    def predict_batch(self, images, conf_threshold=0.25, save_txt=True, save_conf=True):
        """批量预测图像"""
        print(f"开始批量预测 {len(images)} 张图像...")
        print(f"置信度阈值: {conf_threshold}")
        print(f"输出目录: {self.output_dir}")
        
        success_count = 0
        error_count = 0
        
        for i, image_path in enumerate(images, 1):
            try:
                print(f"[{i}/{len(images)}] 预测: {image_path.name}")
                
                # 进行预测
                results = self.model(
                    source=str(image_path),
                    conf=conf_threshold,
                    device=self.device,
                    save=True,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    project=str(self.output_dir),
                    name="predictions",
                    exist_ok=True
                )
                
                success_count += 1
                
                # 打印检测结果摘要
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'masks') and result.masks is not None:
                        mask_count = len(result.masks)
                        print(f"  检测到 {mask_count} 个分割区域")
                    else:
                        print("  未检测到分割区域")
                
            except Exception as e:
                print(f"  预测失败: {e}")
                error_count += 1
        
        print(f"\n批量预测完成!")
        print(f"成功: {success_count} 张")
        print(f"失败: {error_count} 张")
        
        return success_count, error_count
    
    def predict_single(self, image_path, conf_threshold=0.25, save_txt=True, save_conf=True):
        """预测单张图像"""
        print(f"预测单张图像: {image_path}")
        
        results = self.model(
            source=str(image_path),
            conf=conf_threshold,
            device=self.device,
            save=True,
            save_txt=save_txt,
            save_conf=save_conf,
            project=str(self.output_dir),
            name="single_prediction",
            exist_ok=True
        )
        
        print(f"预测结果保存到: {self.output_dir}/single_prediction")
        return results
    
    def organize_results(self):
        """整理预测结果"""
        predictions_dir = self.output_dir / "predictions"
        
        if not predictions_dir.exists():
            print("预测结果目录不存在")
            return
        
        # 创建分类目录
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # 移动图像文件
        for img_file in predictions_dir.glob("*.jpg"):
            shutil.move(str(img_file), str(images_dir / img_file.name))
        
        # 移动标签文件
        for txt_file in predictions_dir.glob("*.txt"):
            shutil.move(str(txt_file), str(labels_dir / txt_file.name))
        
        # 删除空的predictions目录
        if predictions_dir.exists() and not any(predictions_dir.iterdir()):
            predictions_dir.rmdir()
        
        print(f"结果已整理到:")
        print(f"  图像: {images_dir}")
        print(f"  标签: {labels_dir}")

def main():
    # 配置参数
    config = {
        'model_path': 'thermal_segmentation/yolo11_thermal_macos/weights/best.pt',
        'source_dir': './dataset/datasets/thermal_24h/icas',
        'output_dir': './dataset/datasets/thermal_24h/icas_result',
        'conf_threshold': 0.25,  # 置信度阈值
        'save_txt': True,        # 保存标签文件
        'save_conf': True        # 在标签中保存置信度
    }
    
    try:
        # 初始化预测器
        predictor = YOLO11Predictor(
            model_path=config['model_path'],
            source_dir=config['source_dir'],
            output_dir=config['output_dir']
        )
        
        # 加载模型
        predictor.load_model()
        
        # 查找图像
        images = predictor.find_images()
        
        if not images:
            print("源目录中没有找到图像文件")
            return
        
        # 批量预测
        success_count, error_count = predictor.predict_batch(
            images=images,
            conf_threshold=config['conf_threshold'],
            save_txt=config['save_txt'],
            save_conf=config['save_conf']
        )
        
        # 整理结果
        predictor.organize_results()
        
        print(f"\n=== 预测任务完成 ===")
        print(f"处理图像: {len(images)} 张")
        print(f"成功预测: {success_count} 张")
        print(f"预测失败: {error_count} 张")
        print(f"结果保存在: {config['output_dir']}")
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()