"""
天池新闻推荐 - 双塔模型训练脚本
修复内容：
1. user_stats 增加 'last_item' 字段，解决 KeyError
2. 修复了 dataset 和 collate 之间的兼容性
"""

import os
import time
import random
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from model import TwinTowerFull
from dataset import create_dataloaders

# ============================================================
# 数据加载函数 (修正版 V3)
# ============================================================
def load_preprocessed_data_fixed():
    """加载并组装训练所需数据"""
    print("正在加载 pickle 和 npy 文件...")

    with open(USER_CLICKS_PKL, 'rb') as f:
        user_clicks = pickle.load(f)

    # 加载特征矩阵
    category_arr = np.load(CATEGORY_ARR_NPY)
    created_arr = np.load(CREATED_ARR_NPY)
    words_arr = np.load(WORDS_ARR_NPY)
    content_emb_arr = np.load(CONTENT_EMB_ARR_NPY)
    item_probs = np.load(ITEM_SAMPLING_PROBS_NPY)

    # ============================================================
    # 【核心修复】构建 User Stats，必须包含 'last_item'
    # ============================================================
    user_stats = {}
    for uid, clicks in user_clicks.items():
        total = len(clicks)
        if total > 0:
            days = len(set([c['timestamp'] // (3600*24*1000) for c in clicks]))
        else:
            days = 0

        # 获取最后一次历史行为（target 之前的那个点击）
        # 如果只有1次点击，last_item 设为 0 (padding)
        if total >= 2:
            last_item_val = clicks[-2]['item_idx']
        else:
            last_item_val = 0

        user_stats[uid] = {
            'total_clicks': total,
            'active_days': days,
            'last_item': last_item_val  # <--- 这里修复了 KeyError
        }

    # 构建训练样本 (User, Sequence_Index)
    sample_idx_list = []
    for uid, clicks in user_clicks.items():
        if len(clicks) >= 2: # 至少要有1次历史和1次目标
            target_seq_idx = len(clicks) - 1
            sample_idx_list.append((uid, target_seq_idx))

    # 获取最大ID用于模型初始化
    num_categories = int(category_arr.max()) + 1

    # 扫描特征最大值
    max_env = 0
    max_device = 0
    max_os = 0
    max_country = 0
    max_region = 0
    max_referrer = 0

    print("扫描特征最大值...")
    scan_limit = min(len(user_clicks), 5000)
    for i, uid in enumerate(list(user_clicks.keys())[:scan_limit]):
        for c in user_clicks[uid]:
            max_env = max(max_env, c['env'])
            max_device = max(max_device, c['device_group'])
            max_os = max(max_os, c['os'])
            max_country = max(max_country, c['country'])
            max_region = max(max_region, c['region'])
            max_referrer = max(max_referrer, c['referrer'])

    return {
        'sample_idx_list': sample_idx_list,
        'user_clicks': user_clicks,
        'user_stats': user_stats,
        'item_probs': item_probs,
        'category_arr': category_arr,
        'created_arr': created_arr,
        'words_arr': words_arr,
        'content_emb_arr': content_emb_arr,
        'num_categories': num_categories,
        'max_env': max_env + 1,
        'max_device': max_device + 1,
        'max_os': max_os + 1,
        'max_country': max_country + 1,
        'max_region': max_region + 1,
        'max_referrer': max_referrer + 1
    }

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_scores(user_vecs, item_vecs, item_probs, alpha=0.5):
    """
    计算最终得分：cos_sim - alpha * log(p_global)
    """
    B, C, D = item_vecs.shape
    user_vecs_exp = user_vecs.unsqueeze(1).expand(-1, C, -1)  # (B, C, D)

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(user_vecs_exp, item_vecs, dim=-1)  # (B, C)

    # 加入负采样校正项
    scores = cos_sim - alpha * torch.log(item_probs + 1e-12)

    return scores

def train_epoch(model, dataloader, optimizer, device, alpha=0.5):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 预加载特征数组到GPU
    if hasattr(dataloader.dataset, 'feature_arrays'):
        feature_arrays = dataloader.dataset.feature_arrays
        for key in feature_arrays:
            if torch.is_tensor(feature_arrays[key]):
                feature_arrays[key] = feature_arrays[key].to(device)

    # 全局变量引用
    global category_arr_t, created_arr_t, words_arr_t, content_emb_t

    for batch_idx, batch in enumerate(dataloader):
        # 获取batch数据
        B = len(batch['user_id'])

        # 用户特征
        user_id = batch['user_id'].to(device)
        env = batch['env'].to(device)
        device_group = batch['device_group'].to(device)
        os_ = batch['os'].to(device)
        country = batch['country'].to(device)
        region = batch['region'].to(device)
        referrer = batch['referrer'].to(device)
        month = batch['month'].to(device)
        day = batch['day'].to(device)
        hour = batch['hour'].to(device)
        minute = batch['minute'].to(device)

        # 历史序列
        history = batch['history'].to(device)

        # 用户统计
        total_clicks = batch['total_clicks'].to(device).unsqueeze(1)
        active_days = batch['active_days'].to(device).unsqueeze(1)
        user_stats = torch.cat([total_clicks, active_days], dim=1)

        # 最后点击物品 (Target 的前一个物品，作为特征)
        last_items = batch['last_items'].to(device)

        # 候选物品
        candidates = batch['candidates'].to(device)  # (B, C)
        B, C = candidates.shape

        # 获取物品特征
        flat_cand = candidates.reshape(-1)  # (B*C,)

        # 从预加载的数组中获取特征
        if hasattr(dataloader.dataset, 'feature_arrays'):
            fa = dataloader.dataset.feature_arrays
            cat_feat = fa['category_arr'][flat_cand]
            created_feat = fa['created_arr'][flat_cand]
            words_feat = fa['words_arr'][flat_cand]
            content_feat = fa['content_emb_arr'][flat_cand]
        else:
            cat_feat = category_arr_t[flat_cand].to(device)
            created_feat = created_arr_t[flat_cand].to(device)
            words_feat = words_arr_t[flat_cand].to(device)
            content_feat = content_emb_t[flat_cand].to(device)

        num_feat = torch.stack([created_feat, words_feat], dim=1)

        # 最后物品特征
        last_cat = category_arr_t[last_items].to(device)
        last_content = content_emb_t[last_items].to(device)

        # 计算历史统计特征
        B_hist, K = history.shape
        hist_flat = history.reshape(-1)

        try:
            limit = NUM_ITEMS - 1
        except NameError:
            limit = len(category_arr_t) - 1

        hist_flat = torch.clamp(hist_flat, 0, limit)

        if hasattr(dataloader.dataset, 'feature_arrays'):
            hist_created = fa['created_arr'][hist_flat].reshape(B, K)
            hist_words = fa['words_arr'][hist_flat].reshape(B, K)
        else:
            hist_created = created_arr_t[hist_flat].reshape(B, K)
            hist_words = words_arr_t[hist_flat].reshape(B, K)

        # 历史统计特征（平均值）
        hist_num_feats = torch.stack([
            hist_created.mean(dim=1),
            hist_words.mean(dim=1)
        ], dim=1)

        # 前向传播 - 用户塔
        user_vecs = model.forward_user(
            user_id, env, device_group, os_, country, region, referrer,
            month, day, hour, minute, history, hist_num_feats, user_stats,
            last_cat, last_content
        )

        # 前向传播 - 物品塔
        item_vecs_flat = model.forward_item_features(
            flat_cand, cat_feat, num_feat, content_feat
        )
        item_vecs = item_vecs_flat.view(B, C, -1)

        # 计算得分
        item_probs = torch.full((B, C), 1.0/C, device=device)
        scores = compute_scores(user_vecs, item_vecs, item_probs, alpha)

        # listwise损失
        log_probs = F.log_softmax(scores, dim=1)
        loss = -log_probs[:, 0].mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

    return total_loss / num_batches if num_batches > 0 else 0

def export_item_vectors(model, device, output_path):
    """
    导出所有物品向量用于Faiss索引
    """
    print("导出物品向量...")
    model.eval()

    try:
        total_items = NUM_ITEMS
    except NameError:
        total_items = len(category_arr_t)

    with torch.no_grad():
        batch_size = 2048
        item_vectors = torch.zeros(total_items, TOWER_OUTPUT_DIM, device=device)

        for i in range(0, total_items-1, batch_size):
            start_idx = i
            end_idx = min(i + batch_size, total_items-1)

            # 生成 batch_items 索引 (1-based)
            batch_items = torch.arange(start_idx + 1, end_idx + 1, device=device)

            cat_feat = category_arr_t[batch_items].to(device)
            created_feat = created_arr_t[batch_items].to(device)
            words_feat = words_arr_t[batch_items].to(device)
            content_feat = content_emb_t[batch_items].to(device)
            num_feat = torch.stack([created_feat, words_feat], dim=1)

            vecs = model.forward_item_features(batch_items, cat_feat, num_feat, content_feat)

            # 填入结果矩阵
            item_vectors[start_idx+1 : end_idx+1] = vecs

        # 保存到文件
        item_vectors_np = item_vectors.cpu().numpy()
        np.save(output_path, item_vectors_np)

        print(f"物品向量导出完成：{output_path}")
        print(f"向量维度：{item_vectors_np.shape}")

        return item_vectors_np

def main():
    """主训练函数"""
    print("="*50)
    print("天池新闻推荐 - 双塔模型训练")
    print("="*50)

    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"使用设备：{device}")

    print("加载预处理数据...")
    data = load_preprocessed_data_fixed()

    sample_idx_list = data['sample_idx_list']

    user_clicks = data['user_clicks']
    print(user_clicks)
    user_stats = data['user_stats']
    item_probs = data['item_probs']

    global NUM_ITEMS
    NUM_ITEMS = len(data['category_arr'])

    global category_arr_t, created_arr_t, words_arr_t, content_emb_t
    category_arr_t = torch.LongTensor(data['category_arr']).to(device)
    created_arr_t = torch.FloatTensor(data['created_arr']).to(device)
    words_arr_t = torch.FloatTensor(data['words_arr']).to(device)
    content_emb_t = torch.FloatTensor(data['content_emb_arr']).to(device)

    print(f"训练样本数：{len(sample_idx_list)}")
    print(f"用户数：{len(user_clicks)}")
    print(f"物品数：{NUM_ITEMS-1}")

    print("创建数据加载器...")
    dataloader = create_dataloaders(sample_idx_list, user_clicks, user_stats, item_probs)

    print("创建模型...")
    max_uid_in_sample = max([s[0] for s in sample_idx_list]) if sample_idx_list else 0
    print(max_uid_in_sample)
    num_users_model = max(NUM_USERS_TO_USE if NUM_USERS_TO_USE else 0, max_uid_in_sample + 1)

    model = TwinTowerFull(
        num_users=num_users_model,
        num_items=NUM_ITEMS,
        num_categories=data['num_categories'],
        max_env=data['max_env'],
        max_device=data['max_device'],
        max_os=data['max_os'],
        max_country=data['max_country'],
        max_region=data['max_region'],
        max_referrer=data['max_referrer']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("开始训练...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, device, alpha=0.5)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  平均损失：{avg_loss:.6f}")
        print(f"  耗时：{end_time - start_time:.1f}秒")
        print()

    export_item_vectors(model, device, ITEM_VECS_NPY)

    # ✅ 新增
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存到: {MODEL_PATH}")

    print("用户向量将在召回脚本中生成，训练脚本结束。")

if __name__ == "__main__":
    main()