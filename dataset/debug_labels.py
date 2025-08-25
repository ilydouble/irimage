import os
from pathlib import Path

def debug_specific_label():
    """调试具体的标签文件内容"""
    base_dir = Path("./dataset/datasets")
    
    # 找一个具体的标签文件来分析
    labels_dir = base_dir / "thermal_24h" / "icas_result" / "predictions" / "labels"
    
    if not labels_dir.exists():
        print(f"目录不存在: {labels_dir}")
        return
    
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print("没有找到标签文件")
        return
    
    # 分析第一个标签文件
    label_file = label_files[0]
    print(f"分析文件: {label_file}")
    
    with open(label_file, 'r') as f:
        content = f.read().strip()
        print(f"文件内容:\n{content}")
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.strip().split()
                print(f"\n行 {i+1}:")
                print(f"  总部分数: {len(parts)}")
                print(f"  第1部分(class_id): {parts[0]}")
                if len(parts) > 1:
                    print(f"  第2部分: {parts[1]}")
                if len(parts) > 2:
                    print(f"  第3部分: {parts[2]}")
                if len(parts) > 3:
                    print(f"  第4部分: {parts[3]}")
                
                # 尝试解析坐标
                try:
                    # 假设格式是: class_id x1 y1 x2 y2 ...
                    coords = [float(x) for x in parts[1:]]
                    print(f"  坐标数量: {len(coords)}")
                    print(f"  前4个坐标: {coords[:4] if len(coords) >= 4 else coords}")
                except Exception as e:
                    print(f"  坐标解析失败: {e}")

if __name__ == "__main__":
    debug_specific_label()
