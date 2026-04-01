"""
天池新闻推荐 - Item-CF 相似度计算模块
基于物品的协同过滤 + 时间/顺序/创建时间加权
"""

import os
import pickle
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from config import *

def build_user_item_time(user_clicks):
    """
    构建用户-物品-时间索引
    返回: user_item_time_dict = {user1: [(item1, time1), (item2, time2), ...], ...}
    """
    user_item_time_dict = {}
    
    for user_id, clicks in user_clicks.items():
        # 按时间排序
        clicks_sorted = sorted(clicks, key=lambda x: x['timestamp'])
        item_time_list = [(c['item_idx'], c['timestamp']) for c in clicks_sorted]
        user_item_time_dict[user_id] = item_time_list
    
    return user_item_time_dict

def itemcf_sim(user_clicks, item_created_time_dict=None):
    """
    计算物品相似度矩阵（Item-CF）
    支持时间加权、顺序加权、创建时间加权
    """
    print("构建用户-物品-时间索引...")
    user_item_time_dict = build_user_item_time(user_clicks)
    
    print("计算物品相似度...")
    i2i_sim = defaultdict(dict)  # 物品相似度矩阵
    item_cnt = defaultdict(int)   # 物品出现次数
    
    # 遍历每个用户的点击序列
    for user_id, item_time_list in tqdm(user_item_time_dict.items(), desc="计算共现权重"):
        n = len(item_time_list)
        
        for loc1, (i, t_i) in enumerate(item_time_list):
            item_cnt[i] += 1
            
            for loc2, (j, t_j) in enumerate(item_time_list):
                if i == j:
                    continue
                
                # 初始化相似度字典
                i2i_sim.setdefault(i, {})
                i2i_sim[i].setdefault(j, 0)
                
                # 基础共现权重（打压活跃用户）
                base_weight = 1.0 / math.log(1 + n)
                
                # 顺序权重（正向点击权重更高）
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9 ** (abs(loc2 - loc1) - 1)) if abs(loc2 - loc1) > 1 else loc_alpha
                
                # 点击时间权重（时间差越小权重越高）
                click_time_weight = 1.0
                if t_i != t_j:
                    time_diff_hours = abs(t_i - t_j) / (3600 * 1000)  # 转换为小时
                    click_time_weight = math.exp(-0.7 * time_diff_hours / 24)  # 按天衰减
                
                # 文章创建时间权重（创建时间越接近权重越高）
                created_time_weight = 1.0
                if item_created_time_dict and i in item_created_time_dict and j in item_created_time_dict:
                    created_diff_hours = abs(item_created_time_dict[i] - item_created_time_dict[j]) / (3600 * 1000)
                    created_time_weight = math.exp(-0.8 * created_diff_hours / 24)
                
                # 累加共现权重
                i2i_sim[i][j] += base_weight * loc_weight * click_time_weight * created_time_weight
    
    # 归一化（余弦相似度）
    print("归一化相似度矩阵...")
    i2i_sim_ = i2i_sim.copy()
    
    for i, related_items in tqdm(i2i_sim.items(), desc="归一化"):
        for j, wij in related_items.items():
            # 余弦相似度归一化
            norm = math.sqrt(item_cnt[i] * item_cnt[j])
            if norm > 0:
                i2i_sim_[i][j] = wij / norm
    
    return i2i_sim_, dict(item_cnt)

def swing_sim(user_clicks, item_created_time_dict=None, alpha=0.5):
    """
    Swing算法改进版（小圈子惩罚）
    参数alpha控制小圈子惩罚强度
    """
    print("构建用户-物品-时间索引...")
    user_item_time_dict = build_user_item_time(user_clicks)
    
    # 构建物品-用户反向索引
    item_users_dict = defaultdict(set)
    for user_id, item_time_list in user_item_time_dict.items():
        for item_idx, _ in item_time_list:
            item_users_dict[item_idx].add(user_id)
    
    print("计算Swing相似度...")
    item_sim = defaultdict(dict)
    item_cnt = defaultdict(int)
    
    # 遍历每个用户的点击序列
    for user_id, item_time_list in tqdm(user_item_time_dict.items(), desc="计算Swing权重"):
        n = len(item_time_list)
        
        for loc1, (i, t_i) in enumerate(item_time_list):
            item_cnt[i] += 1
            
            for loc2, (j, t_j) in enumerate(item_time_list):
                if i == j:
                    continue
                
                item_sim.setdefault(i, {})
                item_sim[i].setdefault(j, 0)
                
                # 基础共现权重
                base_weight = 1.0 / math.log(1 + n)
                
                # 顺序权重
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9 ** (abs(loc2 - loc1) - 1)) if abs(loc2 - loc1) > 1 else loc_alpha
                
                # 时间权重
                click_time_weight = math.exp(-0.7 * abs(t_i - t_j) / (3600 * 24 * 1000))
                
                # 创建时间权重
                created_time_weight = 1.0
                if item_created_time_dict and i in item_created_time_dict and j in item_created_time_dict:
                    created_time_weight = math.exp(-0.8 * abs(item_created_time_dict[i] - item_created_time_dict[j]) / (3600 * 24 * 1000))
                
                # 小圈子惩罚（Swing核心）
                overlap_users = item_users_dict[i] & item_users_dict[j]
                penalty = 1.0 / (1 + len(overlap_users))  # 小圈子惩罚
                
                # 累加权重
                item_sim[i][j] += base_weight * loc_weight * click_time_weight * created_time_weight * penalty
    
    # 归一化
    print("归一化Swing相似度...")
    item_sim_ = item_sim.copy()
    
    for i, related_items in tqdm(item_sim.items(), desc="归一化"):
        for j, cij in related_items.items():
            norm = math.sqrt(item_cnt[i] * item_cnt[j])
            overlap = len(item_users_dict[i] & item_users_dict[j])
            penalty = 1.0 / (1 + overlap)
            
            if norm > 0:
                item_sim_[i][j] = (cij / norm) * penalty
    
    return item_sim_, dict(item_cnt)

def build_item_topk_click(train_df, k=100):
    """
    获取点击最多的top-k文章（用于冷启动补全）
    """
    topk_click = train_df['click_article_id'].value_counts().index[:k].tolist()
    return topk_click

def get_item_topk_sim(item_sim, k=20):
    """
    为每个物品构建top-k相似物品列表
    返回: item_topk = {item1: [(sim_item1, score1), (sim_item2, score2), ...], ...}
    """
    item_topk = {}
    
    for item_idx, sim_items in tqdm(item_sim.items(), desc="构建topk相似列表"):
        # 按相似度降序排序
        topk_items = sorted(sim_items.items(), key=lambda x: x[1], reverse=True)[:k]
        item_topk[item_idx] = topk_items
    
    return item_topk

def save_similarity_matrix(item_sim, item_cnt, method="itemcf"):
    """
    保存相似度矩阵和物品计数
    """
    save_path = ITEMCF_SIM_PKL if method == "itemcf" else SWING_SIM_PKL
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'item_sim': item_sim,
            'item_cnt': item_cnt
        }, f)
    
    print(f"相似度矩阵已保存到: {save_path}")

def main():
    """主函数：计算Item-CF和Swing相似度"""
    # 加载预处理数据
    print("加载预处理数据...")
    
    with open(USER_CLICKS_PKL, 'rb') as f:
        user_clicks = pickle.load(f)
    
    with open(ARTICLE_MAPPING_PKL, 'rb') as f:
        article_mapping = pickle.load(f)
    
    # 加载文章创建时间字典
    articles_df = pd.read_csv(ARTICLES_PATH)
    item_created_time_dict = dict(zip(articles_df.article_id, articles_df.created_at_ts))
    
    # 将article_id映射到内部idx
    article_id_to_idx = article_mapping['article_id_to_idx']
    created_time_dict = {}
    for article_id, created_time in item_created_time_dict.items():
        if article_id in article_id_to_idx:
            idx = article_id_to_idx[article_id]
            created_time_dict[idx] = created_time
    
    # 1. 计算Item-CF相似度
    print("\n=== 计算Item-CF相似度 ===")
    itemcf_sim_matrix, itemcf_cnt = itemcf_sim(user_clicks, created_time_dict)
    
    # 构建top-k相似列表
    itemcf_topk = get_item_topk_sim(itemcf_sim_matrix, k=20)
    
    # 保存
    save_similarity_matrix(itemcf_sim_matrix, itemcf_cnt, "itemcf")
    with open(ITEMCF_TOPK_PKL, 'wb') as f:
        pickle.dump(itemcf_topk, f)
    
    # 2. 计算Swing相似度
    print("\n=== 计算Swing相似度 ===")
    swing_sim_matrix, swing_cnt = swing_sim(user_clicks, created_time_dict, alpha=0.5)
    
    # 构建top-k相似列表
    swing_topk = get_item_topk_sim(swing_sim_matrix, k=20)
    
    # 保存
    save_similarity_matrix(swing_sim_matrix, swing_cnt, "swing")
    with open(SWING_TOPK_PKL, 'wb') as f:
        pickle.dump(swing_topk, f)
    
    # 3. 构建热门物品列表（用于冷启动补全）
    print("\n=== 构建热门物品列表 ===")
    train_df = pd.read_csv(TRAIN_PATH)
    topk_click = build_item_topk_click(train_df, k=100)
    
    with open(TOPK_CLICK_PKL, 'wb') as f:
        pickle.dump(topk_click, f)
    
    print("所有相似度计算完成！")
    print("输出文件：")
    print(f"  Item-CF相似度: {ITEMCF_SIM_PKL}")
    print(f"  Item-CF topk: {ITEMCF_TOPK_PKL}")
    print(f"  Swing相似度: {SWING_SIM_PKL}")
    print(f"  Swing topk: {SWING_TOPK_PKL}")
    print(f"  热门物品: {TOPK_CLICK_PKL}")

if __name__ == "__main__":
    import pandas as pd  # 在main函数中导入，避免循环依赖
    main()