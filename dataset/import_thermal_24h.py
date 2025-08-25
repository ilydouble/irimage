import os
import sqlite3
from pathlib import Path

class ThermalDataImporter:
    def __init__(self, db_path='../web/database/patientcare.db'):
        self.db_path = db_path
        self.thermal_24h_dir = Path('../2025年合同完整数据/2024热成像数据')
        
    def get_patient_ids(self):
        """从数据库获取thermal_24h为1的患者ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT patient_id FROM patients WHERE thermal_24h = 1 ORDER BY patient_id"
        cursor.execute(query)
        patient_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"从数据库获取到 {len(patient_ids)} 个需要24年热力图的患者ID")
        return patient_ids
    
    def find_thermal_24h_files(self, patient_ids):
        """查找24年热力图文件"""
        found_files = []
        not_found = []
        
        for patient_id in patient_ids:
            # 查找以患者ID命名的文件夹
            patient_folder = self.thermal_24h_dir / patient_id
            
            if patient_folder.exists() and patient_folder.is_dir():
                # 按优先级查找文件：ID+"1".jpg -> ID+"-1".jpg -> ID+"-6".jpg
                file_patterns = [f"{patient_id}1.jpg", f"{patient_id}-1.jpg", f"{patient_id}-6.jpg"]
                target_file = None
                file_name = None
                
                for pattern in file_patterns:
                    test_file = patient_folder / pattern
                    if test_file.exists():
                        target_file = test_file
                        file_name = pattern
                        break
                
                if target_file and target_file.exists():
                    # 计算相对于web目录的路径
                    relative_path = target_file.relative_to(Path('../'))
                    found_files.append({
                        'patient_id': patient_id,
                        'file_name': file_name,
                        'file_path': str(relative_path),
                        'full_path': str(target_file)
                    })
                    print(f"找到文件: {patient_id} -> {relative_path}")
                else:
                    not_found.append(f"{patient_id}: 文件 {patient_id}1.jpg, {patient_id}-1.jpg 和 {patient_id}-6.jpg 都不存在")
            else:
                not_found.append(f"{patient_id}: 文件夹不存在")
        
        print(f"\n找到 {len(found_files)} 个文件")
        print(f"未找到 {len(not_found)} 个文件")
        
        if not_found:
            print("\n未找到的文件:")
            for item in not_found[:10]:  # 只显示前10个
                print(f"  {item}")
            if len(not_found) > 10:
                print(f"  ... 还有 {len(not_found) - 10} 个")
        
        return found_files
    
    def save_files_to_database(self, files_data):
        """将文件信息保存到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文件类型: 0=24年热力图, 1=25年热力图, 2=25年语音
        file_type = 'thermal_24h'
        
        insert_sql = """
        INSERT OR REPLACE INTO files (patient_id, file_type, file_name, file_path, upload_time)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for file_info in files_data:
            try:
                cursor.execute(insert_sql, [
                    file_info['patient_id'],
                    file_type,
                    file_info['file_name'],
                    file_info['file_path']
                ])
                if cursor.rowcount == 0:
                    skipped_count += 1
                else:
                    success_count += 1
            except Exception as e:
                print(f"保存失败 {file_info['patient_id']}: {e}")
                error_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"\n数据库保存结果:")
        print(f"成功保存: {success_count} 个文件记录")
        print(f"保存失败: {error_count} 个文件记录")
        print(f"跳过重复: {skipped_count}")
        
        return success_count, error_count, skipped_count
    
    def import_thermal_24h_data(self):
        """导入24年热力图数据"""
        print("开始导入24年热力图数据...")
        print(f"数据目录: {self.thermal_24h_dir}")
        
        # 检查数据目录是否存在
        if not self.thermal_24h_dir.exists():
            print(f"错误: 数据目录不存在 - {self.thermal_24h_dir}")
            return
        
        # 获取患者ID列表
        patient_ids = self.get_patient_ids()
        
        if not patient_ids:
            print("错误: 数据库中没有患者数据")
            return
        
        # 查找文件
        files_data = self.find_thermal_24h_files(patient_ids)
        
        if not files_data:
            print("错误: 没有找到任何文件")
            return
        
        # 保存到数据库
        success_count, error_count, skipped_count = self.save_files_to_database(files_data)
        
        print(f"\n24年热力图数据导入完成!")
        print(f"处理患者数: {len(patient_ids)}")
        print(f"找到文件数: {len(files_data)}")
        print(f"成功导入: {success_count}")
        print(f"跳过重复: {skipped_count}")
        
        return files_data

if __name__ == "__main__":
    importer = ThermalDataImporter()
    result = importer.import_thermal_24h_data()
