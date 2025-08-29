#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR结果文件提取脚本 - 简化交互版
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime


def find_asr_files_simple(search_dir):
    """简单查找ASR结果文件"""
    search_path = Path(search_dir)
    asr_files = []
    
    if not search_path.exists():
        print(f"❌ 目录不存在: {search_dir}")
        return []
    
    print(f"🔍 搜索目录: {search_dir}")
    
    # 递归查找所有json和txt文件
    for file_path in search_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.json', '.txt']:
            # 检查是否是ASR结果文件
            if is_asr_file_simple(file_path):
                asr_files.append(file_path)
                print(f"  ✓ {file_path.relative_to(search_path)}")
    
    return asr_files


def is_asr_file_simple(file_path):
    """简单判断是否是ASR文件"""
    try:
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 检查关键字段
            return 'text' in data and ('detailed_results' in data or 'raw_response' in data)
        
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(200)
            # 检查是否包含分析报告标识
            return '语音识别详细分析报告' in content or '分句详细信息' in content
    
    except:
        return False
    
    return False


def extract_files_simple(asr_files, output_dir):
    """简单提取文件"""
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    json_dir = output_path / "json_results"
    txt_dir = output_path / "txt_reports"
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for file_path in asr_files:
        try:
            if file_path.suffix.lower() == '.json':
                target_dir = json_dir
            else:
                target_dir = txt_dir
            
            # 处理文件名冲突
            target_file = target_dir / file_path.name
            counter = 1
            while target_file.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                target_file = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # 复制文件
            shutil.copy2(file_path, target_file)
            print(f"📄 复制: {file_path.name} -> {target_file.relative_to(output_path)}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 复制失败: {file_path.name} - {e}")
    
    return success_count


def create_simple_summary(asr_files, output_dir):
    """创建简单汇总"""
    summary_file = Path(output_dir) / "summary.txt"
    
    json_files = [f for f in asr_files if f.suffix.lower() == '.json']
    txt_files = [f for f in asr_files if f.suffix.lower() == '.txt']
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ASR结果文件提取汇总\n")
        f.write("=" * 30 + "\n")
        f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"JSON文件: {len(json_files)} 个\n")
        f.write(f"TXT文件: {len(txt_files)} 个\n")
        f.write(f"总计: {len(asr_files)} 个\n\n")
        
        f.write("文件列表:\n")
        f.write("-" * 20 + "\n")
        for file_path in sorted(asr_files):
            f.write(f"{file_path}\n")
    
    print(f"📋 汇总保存到: {summary_file}")


def main():
    print("🚀 ASR结果文件提取工具 (简化版)")
    print("=" * 40)
    
    # 获取用户输入
    search_dir = input("请输入搜索目录 (默认: dataset/datasets/voice_25h): ").strip()
    if not search_dir:
        search_dir = "dataset/datasets/voice_25h"
    
    output_dir = input("请输入输出目录 (默认: asr_results): ").strip()
    if not output_dir:
        output_dir = "asr_results"
    
    # 查找文件
    print(f"\n🔍 开始搜索...")
    asr_files = find_asr_files_simple(search_dir)
    
    if not asr_files:
        print("❌ 未找到任何ASR结果文件")
        return
    
    print(f"\n📊 找到 {len(asr_files)} 个ASR结果文件")
    
    # 确认提取
    confirm = input(f"\n确认提取到 '{output_dir}' 目录吗? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ 用户取消操作")
        return
    
    # 提取文件
    print(f"\n📁 开始提取...")
    success_count = extract_files_simple(asr_files, output_dir)
    
    # 创建汇总
    create_simple_summary(asr_files, output_dir)
    
    print(f"\n🎉 提取完成!")
    print(f"✅ 成功提取: {success_count} 个文件")
    print(f"📁 结果目录: {output_dir}")
    print(f"  📄 JSON文件: {output_dir}/json_results/")
    print(f"  📊 TXT报告: {output_dir}/txt_reports/")


if __name__ == "__main__":
    main()
