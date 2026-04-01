"""
天池新闻推荐 - Faiss向量索引构建与召回
支持内积相似度搜索 + 命中率评估
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm

from config import *
from model import TwinTowerFull
from preprocess import *
# from preprocess import load_preprocessed_data

with open(ARTICLE_MAPPING_PKL, 'rb') as f:
    mapping = pickle.load(f)

NUM_ITEMS = len(mapping['article_id_to_idx']) + 1

def build_faiss_index(item_vectors_path, output_path):
    """
    构建Faiss索引
    Args:
        item_vectors_path: 物品向量文件路径
        output_path: 索引保存路径
    Returns:
        index: Faiss索引对象
    """
    print("构建Faiss索引...")
    
    # 加载物品向量
    item_vectors = np.load(item_vectors_path).astype('float32')
    
    # 移除padding向量（索引0）
    if len(item_vectors) > NUM_ITEMS - 1:
        item_vectors = item_vectors[1:]  # 跳过padding
    
    print(f"物品向量形状: {item_vectors.shape}")
    
    # L2归一化（用于余弦相似度）
    faiss.normalize_L2(item_vectors)
    
    # 创建内积索引（归一化后内积=余弦相似度）
    d = item_vectors.shape[1]  # 向量维度
    index = faiss.IndexFlatIP(d)
    
    # 添加向量到索引
    index.add(item_vectors)
    
    print(f"Faiss索引构建完成，总物品数: {index.ntotal}")
    
    # 保存索引
    faiss.write_index(index, output_path)
    print(f"索引已保存至: {output_path}")
    
    return index

def generate_user_vectors(model, device, user_range=None):
    """
    生成用户向量
    Args:
        model: 训练好的模型
        device: 计算设备
        user_range: 用户ID范围，默认使用配置中的NUM_USERS_TO_USE
    Returns:
        user_vectors: 用户向量数组 (N, D)
        user_ids: 用户ID列表
    """
    print("生成用户向量...")
    
    if user_range is None:
        user_range = range(NUM_USERS_TO_USE)
    
    # 加载预处理数据
    data = load_preprocessed_data()
    user_clicks = data['user_clicks']
    user_stats = data['user_stats']
    
    # 加载特征数组
    category_arr = data['category_arr']
    content_emb_arr = data['content_emb_arr']
    
    # 转换为tensor
    category_arr_t = torch.LongTensor(category_arr).to(device)
    content_emb_t = torch.FloatTensor(content_emb_arr).to(device)
    
    model.eval()
    user_vectors = []
    user_ids = []
    
    with torch.no_grad():
        for user_id in tqdm(user_range, desc="生成用户向量"):
            if user_id not in user_clicks:
                continue
                
            clicks = user_clicks[user_id]
            if len(clicks) < 1:
                continue
            
            # 历史序列（不包含最后一次）
            history_ids = [c['item_idx'] for c in clicks[:-1]]
            if len(history_ids) < HISTORY_POOL_K:
                pad_len = HISTORY_POOL_K - len(history_ids)
                history_ids = [0] * pad_len + history_ids
            else:
                history_ids = history_ids[-HISTORY_POOL_K:]
            
            history_tensor = torch.LongTensor([history_ids]).to(device)
            
            # 最后点击物品
            last_item_idx = clicks[-1]['item_idx']
            last_cat = torch.LongTensor([category_arr[last_item_idx]]).to(device)
            last_content = torch.FloatTensor(np.array([content_emb_arr[last_item_idx]])).to(device)
            
            # 用户上下文特征
            last_click = clicks[-1]
            env = torch.LongTensor([last_click['env']]).to(device)
            device_group = torch.LongTensor([last_click['device_group']]).to(device)
            os_ = torch.LongTensor([last_click['os']]).to(device)
            country = torch.LongTensor([last_click['country']]).to(device)
            region = torch.LongTensor([last_click['region']]).to(device)
            referrer = torch.LongTensor([last_click['referrer']]).to(device)
            
            # 时间特征
            dt = pd.to_datetime(last_click['timestamp'], unit='ms')
            month = torch.LongTensor([dt.month]).to(device)
            day = torch.LongTensor([dt.day]).to(device)
            hour = torch.LongTensor([dt.hour]).to(device)
            minute = torch.LongTensor([dt.minute]).to(device)
            
            # 用户统计特征
            stats = user_stats[user_id]
            total_clicks = torch.FloatTensor([[stats['total_clicks'], stats['active_days']]]).to(device)
            
            # 历史统计特征（简化）
            hist_num_feats = total_clicks.clone()
            
            # 生成用户向量
            user_vec = model.forward_user(
                torch.LongTensor([user_id]).to(device),
                env, device_group, os_, country, region, referrer,
                month, day, hour, minute,
                history_tensor, hist_num_feats, total_clicks,
                last_cat, last_content
            )
            
            user_vectors.append(user_vec.cpu().numpy())
            user_ids.append(user_id)
    
    # 合并为矩阵
    user_vectors = np.vstack(user_vectors).astype('float32')
    
    # L2归一化
    faiss.normalize_L2(user_vectors)
    
    print(f"用户向量生成完成，形状: {user_vectors.shape}")
    
    return user_vectors, user_ids

def evaluate_recall(user_vectors, user_ids, index, user_clicks, topk=30):
    """
    评估召回效果
    Args:
        user_vectors: 用户向量 (N, D)
        user_ids: 用户ID列表
        index: Faiss索引
        user_clicks: 用户点击数据
        topk: 召回topk
    Returns:
        hit_rate: 命中率
        hit_users: 命中用户列表
    """
    print(f"评估Top-{topk}召回效果...")
    
    # 搜索topk相似物品
    D, I = index.search(user_vectors, topk)  # I: (N, topk)
    
    hit_count = 0
    hit_users = []
    
    for i, user_id in enumerate(tqdm(user_ids, desc="评估召回")):
        if user_id not in user_clicks:
            continue
            
        clicks = user_clicks[user_id]
        if len(clicks) < 1:
            continue
        
        # 目标物品（最后一次点击）
        target_item = clicks[-1]['item_idx']
        
        # 召回的物品列表（需要+1对齐，因为Faiss索引从0开始）
        recalled_items = I[i] + 1
        
        # 检查是否命中
        if target_item in recalled_items:
            hit_count += 1
            hit_users.append(user_id)
    
    hit_rate = hit_count / len(user_ids) if len(user_ids) > 0 else 0
    
    print(f"Top-{topk} 命中率: {hit_rate:.4f}")
    print(f"命中用户数量: {hit_count}")
    print(f"总评估用户数: {len(user_ids)}")
    if len(hit_users) > 0:
        print(f"命中用户ID示例（前10个）: {hit_users[:10]}")
    
    return hit_rate, hit_users

def main():
    """主函数"""
    print("="*50)
    print("天池新闻推荐 - Faiss索引构建与评估")
    print("="*50)
    
    # 创建设备
    device = torch.device(DEVICE)
    print(f"使用设备: {device}")


    # 检查文件是否存在
    if not os.path.exists(ITEM_VECS_NPY):
        print(f"错误: 物品向量文件不存在 - {ITEM_VECS_NPY}")
        print("请先运行 train.py 训练模型并导出物品向量")
        return
    
    # 构建Faiss索引
    if not os.path.exists(FAISS_INDEX_PATH):
        index = build_faiss_index(ITEM_VECS_NPY, FAISS_INDEX_PATH)
    else:
        print("加载已有Faiss索引...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"索引加载完成，总物品数: {index.ntotal}")
    
    # 加载模型（用于生成用户向量）
    print("加载训练好的模型...")
    data = load_preprocessed_data()
    
    model = TwinTowerFull(
        num_users=NUM_USERS_TO_USE,
        num_items=NUM_ITEMS,
        num_categories=data['num_categories'],
        # max_env=data['max_env'],
        max_env=data['max_env'],
        max_device=5,
        max_os=data['max_os'],
        max_country=data['max_country'],
        max_region=data['max_region'],
        max_referrer=data['max_referrer']
    ).to(device)
    
    # 加载模型权重（假设保存在MODEL_PATH）
    state_dict = torch.load('output/model.pth')  # 替换为你的路径
    print(state_dict['device_emb.weight'].shape)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"模型权重加载完成: {MODEL_PATH}")
    else:
        print("警告: 未找到模型权重文件，使用随机初始化权重")
    
    # 生成用户向量
    user_vectors, user_ids = generate_user_vectors(model, device)
    
    # 保存用户向量
    np.save(USER_VECS_NPY, user_vectors)
    print(f"用户向量已保存: {USER_VECS_NPY}")
    
    # 评估召回效果
    user_clicks = data['user_clicks']
    hit_rate, hit_users = evaluate_recall(user_vectors, user_ids, index, user_clicks, topk=30)
    
    # 保存评估结果
    result = {
        'hit_rate': hit_rate,
        'hit_count': len(hit_users),
        'total_users': len(user_ids),
        'hit_users': hit_users[:100]  # 保存前100个命中用户
    }
    
    with open(EVAL_RESULT_PATH, 'wb') as f:
        pickle.dump(result, f)
    
    print("="*50)
    print("评估完成！")
    print(f"结果已保存至: {EVAL_RESULT_PATH}")

if __name__ == "__main__":
    main()