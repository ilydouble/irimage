#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目结构整理脚本
将所有结果文件整理到统一的 results 文件夹下
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime


def create_results_structure():
    """创建统一的结果文件夹结构"""
    base_dir = Path("results")
    
    # 创建主要结果目录
    directories = [
        "asr_results",           # ASR语音识别结果
        "training_results",      # 模型训练结果
        "analysis_results",      # 分析结果
        "extracted_results",     # 提取的结果文件
        "oss_backups",          # OSS备份文件
        "logs"                  # 日志文件
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")
    
    return base_dir


def move_asr_results():
    """移动ASR识别结果文件"""
    print("\n📁 整理ASR识别结果...")
    
    asr_dir = Path("results/asr_results")
    moved_count = 0
    
    # 移动根目录下的识别结果文件
    for pattern in ["recognition_results_*.json", "recognition_results_*.txt"]:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                target_path = asr_dir / file_path.name
                shutil.move(str(file_path), str(target_path))
                print(f"  📄 移动: {file_path.name}")
                moved_count += 1
    
    # 移动dataset目录下的识别结果
    dataset_voice_dir = Path("dataset/datasets/voice_25h")
    if dataset_voice_dir.exists():
        for json_file in dataset_voice_dir.rglob("*.json"):
            if "recognition" in json_file.name or any(keyword in json_file.name for keyword in ["_analysis", "_result"]):
                target_path = asr_dir / json_file.name
                if not target_path.exists():
                    shutil.copy2(str(json_file), str(target_path))
                    print(f"  📄 复制: {json_file.name}")
                    moved_count += 1
        
        for txt_file in dataset_voice_dir.rglob("*.txt"):
            if "analysis" in txt_file.name or "recognition" in txt_file.name:
                target_path = asr_dir / txt_file.name
                if not target_path.exists():
                    shutil.copy2(str(txt_file), str(target_path))
                    print(f"  📄 复制: {txt_file.name}")
                    moved_count += 1
    
    print(f"📊 ASR结果文件: 处理了 {moved_count} 个文件")
    return moved_count


def move_training_results():
    """移动训练结果文件夹"""
    print("\n🎯 整理训练结果...")
    
    training_dir = Path("results/training_results")
    moved_count = 0
    
    # 移动model目录下的结果文件夹
    model_dir = Path("model")
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir() and "results" in item.name:
                target_path = training_dir / item.name
                if not target_path.exists():
                    shutil.move(str(item), str(target_path))
                    print(f"  📁 移动: {item.name}")
                    moved_count += 1
    
    # 移动其他训练相关文件
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file() and "yolo" in file_path.name.lower():
                target_path = training_dir / file_path.name
                if not target_path.exists():
                    shutil.move(str(file_path), str(target_path))
                    print(f"  📄 移动: {file_path.name}")
                    moved_count += 1
    
    print(f"📊 训练结果: 处理了 {moved_count} 个项目")
    return moved_count


def move_analysis_results():
    """移动分析结果文件"""
    print("\n📈 整理分析结果...")
    
    analysis_dir = Path("results/analysis_results")
    moved_count = 0
    
    # 移动分析相关的图片和文件
    for pattern in ["*_test.png", "*_analysis.png", "*_plot.png", "*.html"]:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                target_path = analysis_dir / file_path.name
                if not target_path.exists():
                    shutil.move(str(file_path), str(target_path))
                    print(f"  📄 移动: {file_path.name}")
                    moved_count += 1
    
    # 移动catboost_info
    catboost_dir = Path("catboost_info")
    if catboost_dir.exists():
        target_path = analysis_dir / "catboost_info"
        if not target_path.exists():
            shutil.move(str(catboost_dir), str(target_path))
            print(f"  📁 移动: catboost_info")
            moved_count += 1
    
    print(f"📊 分析结果: 处理了 {moved_count} 个项目")
    return moved_count


def move_extracted_files():
    """移动提取的文件"""
    print("\n📦 整理提取的文件...")
    
    extracted_dir = Path("results/extracted_results")
    moved_count = 0
    
    # 移动可能的提取结果目录
    for pattern in ["asr_results*", "extracted_*", "*_extracted"]:
        for item in Path(".").glob(pattern):
            if item.is_dir():
                target_path = extracted_dir / item.name
                if not target_path.exists():
                    shutil.move(str(item), str(target_path))
                    print(f"  📁 移动: {item.name}")
                    moved_count += 1
    
    print(f"📊 提取文件: 处理了 {moved_count} 个项目")
    return moved_count


def update_script_paths():
    """更新脚本中的路径引用"""
    print("\n🔧 更新脚本路径引用...")
    
    # 需要更新的脚本文件
    scripts_to_update = [
        "model/voice_asr.py",
        "extract_asr_simple.py",
        "extract_asr_results.py"
    ]
    
    updated_count = 0
    
    for script_path in scripts_to_update:
        script_file = Path(script_path)
        if script_file.exists():
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 更新默认输出路径
                original_content = content
                content = content.replace(
                    'default="asr_results"',
                    'default="results/extracted_results/asr_results"'
                )
                content = content.replace(
                    'output_dir = "asr_results"',
                    'output_dir = "results/extracted_results/asr_results"'
                )
                
                if content != original_content:
                    with open(script_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  📝 更新: {script_path}")
                    updated_count += 1
                    
            except Exception as e:
                print(f"  ⚠️ 更新失败: {script_path} - {e}")
    
    print(f"📊 脚本更新: 处理了 {updated_count} 个文件")
    return updated_count


def create_summary_report():
    """创建整理汇总报告"""
    print("\n📋 创建整理汇总报告...")
    
    results_dir = Path("results")
    report_file = results_dir / "organization_summary.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("项目结构整理汇总报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"整理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 统计各目录的文件数量
            for subdir in results_dir.iterdir():
                if subdir.is_dir():
                    file_count = len(list(subdir.rglob("*")))
                    f.write(f"{subdir.name}: {file_count} 个项目\n")
            
            f.write(f"\n目录结构:\n")
            f.write("-" * 30 + "\n")
            
            # 递归显示目录结构
            def write_tree(path, prefix="", file_handle=f):
                items = sorted(path.iterdir())
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    file_handle.write(f"{prefix}{current_prefix}{item.name}\n")
                    
                    if item.is_dir() and len(list(item.iterdir())) > 0:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        write_tree(item, next_prefix, file_handle)
            
            write_tree(results_dir)
        
        print(f"📋 汇总报告已保存: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 创建汇总报告失败: {e}")


def main():
    print("🚀 开始整理项目结构...")
    print("=" * 50)
    
    # 创建结果目录结构
    create_results_structure()
    
    # 移动各类文件
    asr_count = move_asr_results()
    training_count = move_training_results()
    analysis_count = move_analysis_results()
    extracted_count = move_extracted_files()
    
    # 更新脚本路径
    script_count = update_script_paths()
    
    # 创建汇总报告
    create_summary_report()
    
    # 显示总结
    print("\n🎉 项目结构整理完成!")
    print("=" * 50)
    print(f"📁 ASR结果文件: {asr_count} 个")
    print(f"🎯 训练结果: {training_count} 个")
    print(f"📈 分析结果: {analysis_count} 个")
    print(f"📦 提取文件: {extracted_count} 个")
    print(f"📝 脚本更新: {script_count} 个")
    print(f"\n📋 所有结果文件已整理到 'results/' 目录下")
    print(f"📄 查看详细报告: results/organization_summary.txt")


if __name__ == "__main__":
    main()
