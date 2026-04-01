#!/usr/bin/env python3
"""
天池新闻推荐 - 一键完整pipeline
顺序执行: 数据预处理 → ItemCF相似度 → 双塔训练 → Faiss索引 → 评估
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"开始: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {description} 失败")
        print(f"错误输出: {result.stderr}")
        return False
    
    elapsed = time.time() - start_time
    print(f"完成: {description} (耗时: {elapsed:.1f}s)")
    if result.stdout:
        print(f"输出: {result.stdout[:200]}...")  # 只显示前200字符
    
    return True

def check_data_files():
    """检查必要的数据文件是否存在"""
    required_files = [
        TRAIN_PATH,
        ARTICLES_PATH,
        ARTICLES_EMB_PATH
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("缺少必要的数据文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请先运行: ./download_data.sh 下载数据集")
        return False
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("天池新闻推荐 - 完整Pipeline")
    print("="*60)
    print(f"用户规模: {NUM_USERS_TO_USE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"设备: {DEVICE}")
    print("="*60)
    
    # 检查数据文件
    if not check_data_files():
        return 1
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 步骤1: 数据预处理
    if not run_command(f"python preprocess.py", "数据预处理"):
        return 1
    
    # 步骤2: ItemCF相似度计算
    if not run_command(f"python itemcf.py", "ItemCF相似度计算"):
        return 1
    
    # 步骤3: 双塔模型训练
    if not run_command(f"python train.py", "双塔模型训练"):
        return 1
    
    # 步骤4: Faiss索引构建与评估
    if not run_command(f"python faiss_index.py", "Faiss索引构建与评估"):
        return 1
    
    # 显示最终结果
    print("\n" + "="*60)
    print("Pipeline 执行完成！")
    print("="*60)
    
    # 检查评估结果
    if os.path.exists(EVAL_RESULT_PATH):
        import pickle
        with open(EVAL_RESULT_PATH, 'rb') as f:
            result = pickle.load(f)
        
        print("最终评估结果:")
        print(f"  Top-30 命中率: {result['hit_rate']:.4f}")
        print(f"  命中用户数: {result['hit_count']}")
        print(f"  总评估用户数: {result['total_users']}")
        print(f"  评估结果文件: {EVAL_RESULT_PATH}")
    
    # 列出所有输出文件
    print(f"\n输出文件列表:")
    output_files = [
        ARTICLE_ID_TO_IDX_PKL,
        USER_CLICKS_PKL,
        ITEM_CLICK_COUNT_PKL,
        ITEM_SAMPLING_PROBS_NPY,
        ITEMCF_SIM_PKL,
        SWING_SIM_PKL,
        ITEMCF_TOPK_PKL,
        SWING_TOPK_PKL,
        TOPK_CLICK_PKL,
        MODEL_PATH,
        ITEM_VECS_NPY,
        USER_VECS_NPY,
        FAISS_INDEX_PATH,
        EVAL_RESULT_PATH
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  {os.path.basename(file_path):<25} {size:>8.1f} MB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())