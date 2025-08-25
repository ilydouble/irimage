import os
import sqlite3
import shutil
from pathlib import Path
import json

class DatasetBuilder:
    def __init__(self, db_path='../web/database/patientcare.db', source_data_dir='../data'):
        self.db_path = db_path
        self.source_data_dir = Path(source_data_dir)
        self.dataset_dir = Path('./datasets')
        
        # 创建数据集目录
        self.thermal_24h_dir = self.dataset_dir / 'thermal_24h'
        self.thermal_25h_dir = self.dataset_dir / 'thermal_25h'
        self.voice_25h_dir = self.dataset_dir / 'voice_25h'
        
        self._create_directories()
    
    def _create_directories(self):
        """创建数据集目录结构"""
        for dataset_dir in [self.thermal_24h_dir, self.thermal_25h_dir, self.voice_25h_dir]:
            # 为每个数据集创建ICAS和非ICAS分类文件夹
            (dataset_dir / 'icas').mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'non_icas').mkdir(parents=True, exist_ok=True)
    
    def get_files_with_patient_info(self):
        """从files表获取文件信息，关联患者ICAS状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT f.id, f.patient_id, f.file_type, f.file_path, f.file_name,
               p.name, p.has_icas, p.icas_grade
        FROM files f
        JOIN patients p ON f.patient_id = p.patient_id
        WHERE f.file_path IS NOT NULL
        ORDER BY f.file_type, f.patient_id
        """
        
        cursor.execute(query)
        files_data = cursor.fetchall()
        conn.close()
        
        return files_data
    
    def move_files_to_datasets(self, files_data):
        """将文件移动到对应的数据集文件夹"""
        results = {'thermal_24h': {'icas': 0, 'non_icas': 0},
                  'thermal_25h': {'icas': 0, 'non_icas': 0},
                  'voice_25h': {'icas': 0, 'non_icas': 0}}
        
        for file_id, patient_id, file_type, file_path, file_name, patient_name, has_icas, icas_grade in files_data:
            # 跳过不需要的文件类型
            if file_type not in ['thermal_24h', 'thermal_25h', 'voice_25h']:
                continue
            
            # 确定分类：ICAS阳性或阴性
            category = 'icas' if has_icas else 'non_icas'
            
            # 确定目标目录
            dataset_dirs = {
                'thermal_24h': self.thermal_24h_dir,
                'thermal_25h': self.thermal_25h_dir,
                'voice_25h': self.voice_25h_dir
            }
            target_dir = dataset_dirs[file_type] / category
            
            # 构建源文件完整路径
            source_file = Path('../') / file_path.lstrip('/')
            
            if source_file.exists():
                # 创建新文件名：患者ID_原文件名
                new_filename = f"{patient_id}_{file_name}"
                target_file = target_dir / new_filename
                
                try:
                    shutil.copy2(source_file, target_file)
                    results[file_type][category] += 1
                    print(f"[{file_type}] {category}: {source_file.name} -> {new_filename}")
                except Exception as e:
                    print(f"复制失败 {source_file}: {e}")
            else:
                print(f"源文件不存在: {source_file}")
        
        return results
    
    def generate_dataset_summary(self, results):
        """生成数据集摘要"""
        summary = {
            'thermal_24h': results['thermal_24h'],
            'thermal_25h': results['thermal_25h'],
            'voice_25h': results['voice_25h'],
            'total_files': sum(sum(r.values()) for r in results.values())
        }
        
        # 保存摘要到JSON文件
        summary_file = self.dataset_dir / 'dataset_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n数据集摘要已保存到: {summary_file}")
        print(f"总文件数: {summary['total_files']}")
        
        return summary
    
    def build_all_datasets(self):
        """构建所有数据集"""
        print("开始构建数据集...")
        
        # 从files表获取文件信息和患者ICAS状态
        files_data = self.get_files_with_patient_info()
        print(f"从数据库获取到 {len(files_data)} 个文件记录")
        
        # 移动文件到数据集文件夹
        results = self.move_files_to_datasets(files_data)
        
        # 打印统计信息
        for dataset_type, counts in results.items():
            print(f"{dataset_type}: ICAS={counts['icas']}, 非ICAS={counts['non_icas']}")
        
        # 生成摘要
        summary = self.generate_dataset_summary(results)
        
        print("\n数据集构建完成！")
        return summary

if __name__ == "__main__":
    builder = DatasetBuilder()
    summary = builder.build_all_datasets()
