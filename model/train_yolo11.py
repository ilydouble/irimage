import os
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml

class YOLO11Trainer:
    def __init__(self, 
                 model_name="yolo11n-seg.pt",
                 dataset_yaml="./dataset/datasets/thermal_24h_yolo/dataset.yaml",
                 project_name="thermal_segmentation",
                 experiment_name="yolo11_thermal"):
        
        self.model_name = model_name
        self.dataset_yaml = dataset_yaml
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # 检查数据集配置文件是否存在
        if not Path(self.dataset_yaml).exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.dataset_yaml}")
        
        # macOS设备检测和配置
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
            print("使用CPU训练")
            return 'cpu'
    
    def load_model(self):
        """加载YOLO11模型"""
        print(f"加载模型: {self.model_name}")
        
        # 如果是预训练模型，会自动下载
        self.model = YOLO(self.model_name)
        
        print(f"模型架构: {self.model_name}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
    def verify_dataset(self):
        """验证数据集配置"""
        print("验证数据集配置...")
        
        with open(self.dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"数据集路径: {dataset_config['path']}")
        print(f"类别数量: {dataset_config['nc']}")
        print(f"类别名称: {dataset_config['names']}")
        
        # 检查训练和验证目录
        dataset_path = Path(dataset_config['path'])
        train_images = dataset_path / dataset_config['train']
        val_images = dataset_path / dataset_config['val']
        
        train_count = len(list(train_images.glob('*.jpg'))) if train_images.exists() else 0
        val_count = len(list(val_images.glob('*.jpg'))) if val_images.exists() else 0
        
        print(f"训练图像数量: {train_count}")
        print(f"验证图像数量: {val_count}")
        
        if train_count == 0:
            raise ValueError("训练集为空，请检查数据集路径")
        
        return dataset_config
    
    def train(self, 
              epochs=100,
              imgsz=640,
              batch_size=8,  # macOS降低批次大小
              lr0=0.01,
              patience=50,
              save_period=10,
              workers=4):  # macOS降低worker数量
        """训练YOLO11模型 - macOS优化版"""
        
        # macOS特定的批次大小调整
        if self.device == 'mps':
            batch_size = min(batch_size, 8)  # MPS建议较小批次
            workers = min(workers, 4)  # 减少worker数量
            print("MPS设备检测到，调整批次大小和worker数量")
        elif self.device == 'cpu':
            batch_size = min(batch_size, 4)  # CPU使用更小批次
            workers = min(workers, 2)
            print("CPU设备检测到，使用较小批次大小")
        
        print("开始训练YOLO11模型...")
        print(f"训练参数:")
        print(f"  - 训练轮数: {epochs}")
        print(f"  - 图像尺寸: {imgsz}")
        print(f"  - 批次大小: {batch_size}")
        print(f"  - 学习率: {lr0}")
        print(f"  - 早停耐心: {patience}")
        print(f"  - 保存周期: {save_period}")
        print(f"  - Worker数量: {workers}")
        
        # 训练模型
        results = self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            patience=patience,
            save_period=save_period,
            workers=workers,
            project=self.project_name,
            name=self.experiment_name,
            device=self.device,
            # 分割任务特定参数
            task='segment',
            # macOS优化的数据增强参数（较温和）
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.05,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.8,  # 降低mosaic强度
            mixup=0.0,
            copy_paste=0.0,
            # macOS内存优化
            amp=False if self.device == 'mps' else True,  # MPS可能不支持AMP
        )
        
        print("训练完成!")
        return results
    
    def evaluate(self, model_path=None):
        """评估模型性能"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        print("开始模型评估...")
        results = model.val(data=self.dataset_yaml, device=self.device)
        
        print("评估结果:")
        if hasattr(results, 'box') and results.box:
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def predict_sample(self, image_path, model_path=None, save_dir="./predictions"):
        """对单张图像进行预测"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        print(f"对图像进行预测: {image_path}")
        
        results = model(image_path, save=True, project=save_dir, device=self.device)
        
        print(f"预测结果保存到: {save_dir}")
        return results

def main():
    # macOS优化的训练配置
    config = {
        'model_name': 'yolo11n-seg.pt',  # 使用分割模型
        'dataset_yaml': './dataset/datasets/thermal_24h_yolo/dataset.yaml',
        'project_name': 'thermal_segmentation',
        'experiment_name': 'yolo11_thermal_macos',
        'epochs': 50,  # macOS可能需要更少轮数进行测试
        'imgsz': 640,
        'batch_size': 8,  # 较小批次适合macOS
        'lr0': 0.01,
        'patience': 30,  # 较短耐心值
        'workers': 4  # 适合macOS的worker数量
    }
    
    try:
        # 初始化训练器
        trainer = YOLO11Trainer(
            model_name=config['model_name'],
            dataset_yaml=config['dataset_yaml'],
            project_name=config['project_name'],
            experiment_name=config['experiment_name']
        )
        
        # 加载模型
        trainer.load_model()
        
        # 验证数据集
        dataset_config = trainer.verify_dataset()
        
        # 开始训练
        results = trainer.train(
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch_size=config['batch_size'],
            lr0=config['lr0'],
            patience=config['patience'],
            workers=config['workers']
        )
        
        # 训练完成后评估
        print("\n开始最终评估...")
        eval_results = trainer.evaluate()
        
        # 保存训练配置
        import json
        config_path = f"{config['project_name']}/{config['experiment_name']}/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n训练完成! 模型保存在: {config['project_name']}/{config['experiment_name']}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()