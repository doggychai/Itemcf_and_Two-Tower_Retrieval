"""
天池新闻推荐 - 双塔深度模型
TwinTower + DCNv2 backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import *

class DCNv2Block(nn.Module):
    """
    DCNv2 (Deep & Cross Network v2) 模块
    用于学习高阶特征交互
    """
    def __init__(self, input_dim, cross_layers=2, deep_layers=[256, 128]):
        super().__init__()
        self.input_dim = input_dim
        
        # Cross Network部分
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(cross_layers)
        ])
        
        # Deep Network部分
        deep_modules = []
        in_dim = input_dim
        for h in deep_layers:
            deep_modules.append(nn.Linear(in_dim, h))
            deep_modules.append(nn.ReLU())
            in_dim = h
        self.deep = nn.Sequential(*deep_modules)
        
        # 输出投影
        self.post_proj = nn.Linear(input_dim + in_dim, TOWER_OUTPUT_DIM)
        
    def forward(self, x):
        # Cross Network
        x0 = x
        x_cross = x
        for layer in self.cross_layers:
            x_cross = x_cross + x0 * layer(x_cross)
        
        # Deep Network
        x_deep = self.deep(x0)
        
        # 拼接cross和deep特征
        x_cat = torch.cat([x_cross, x_deep], dim=-1)
        out = self.post_proj(x_cat)
        
        return out

class TwinTowerFull(nn.Module):
    """
    双塔模型：User Tower + Item Tower
    支持完整的用户和物品特征
    """
    def __init__(self, num_users, num_items, num_categories, 
                 max_env, max_device, max_os, max_country, max_region, max_referrer):
        super().__init__()
        
        # 共享的item embedding
        self.item_id_emb = nn.Embedding(num_items, ITEM_ID_EMB_DIM, padding_idx=0)
        
        # User Tower embeddings
        self.user_id_emb = nn.Embedding(num_users + 1, USER_ID_EMB_DIM)
        self.env_emb = nn.Embedding(max_env + 1, OTHER_CAT_EMB_DIM)
        self.device_emb = nn.Embedding(max_device + 1, OTHER_CAT_EMB_DIM)
        self.os_emb = nn.Embedding(max_os + 1, OTHER_CAT_EMB_DIM)
        self.country_emb = nn.Embedding(max_country + 1, OTHER_CAT_EMB_DIM)
        self.region_emb = nn.Embedding(max_region + 1, OTHER_CAT_EMB_DIM)
        self.referrer_emb = nn.Embedding(max_referrer + 1, OTHER_CAT_EMB_DIM)
        
        # 时间embeddings
        self.month_emb = nn.Embedding(13, TIME_EMB_DIM)
        self.day_emb = nn.Embedding(32, TIME_EMB_DIM)
        self.hour_emb = nn.Embedding(24, TIME_EMB_DIM)
        self.minute_emb = nn.Embedding(60, TIME_EMB_DIM)
        
        # 数值特征投影
        self.hist_num_proj = nn.Linear(2, 16)  # 历史统计特征
        self.user_stat_proj = nn.Linear(2, 16)  # 用户统计特征
        self.content_proj = nn.Linear(250, CONTENT_PROJ_DIM // 2)  # 内容embedding维度250
        self.item_num_proj = nn.Linear(2, CONTENT_PROJ_DIM // 2)
        
        # 类别embedding
        self.category_emb = nn.Embedding(num_categories + 1, CAT_EMB_DIM)
        
        # User Tower backbone
        user_input_dim = (USER_ID_EMB_DIM + 6 * OTHER_CAT_EMB_DIM + 4 * TIME_EMB_DIM + 
                         ITEM_ID_EMB_DIM + 16 + 16 + CAT_EMB_DIM + (CONTENT_PROJ_DIM // 2))
        self.user_tower = DCNv2Block(user_input_dim, cross_layers=2, deep_layers=[256, 128])
        
        # Item Tower backbone
        item_input_dim = (ITEM_ID_EMB_DIM + CAT_EMB_DIM + (CONTENT_PROJ_DIM // 2) + (CONTENT_PROJ_DIM // 2))
        self.item_tower = DCNv2Block(item_input_dim, cross_layers=2, deep_layers=[256, 128])
        
    def forward_user(self, user_id, env, device_group, os_, country, region, referrer,
                    month, day, hour, minute, history_ids, hist_num_feats, user_stats,
                    last_cat, last_content):
        """
        User Tower前向传播
        参数:
            user_id: (B,)
            env, device_group, os_, country, region, referrer: (B,)
            month, day, hour, minute: (B,)
            history_ids: (B, K) 历史点击物品
            hist_num_feats: (B, 2) 历史数值特征（创建时间、词数均值）
            user_stats: (B, 2) 用户统计（总点击数、活跃天数）
            last_cat: (B,) 最后一次点击的类别
            last_content: (B, content_dim) 最后一次点击的内容embedding
        返回:
            user_vec: (B, TOWER_OUTPUT_DIM) L2归一化
        """
        # 基础embeddings
        u_e = self.user_id_emb(user_id)
        env_e = self.env_emb(env)
        dev_e = self.device_emb(device_group)
        os_e = self.os_emb(os_)
        country_e = self.country_emb(country)
        region_e = self.region_emb(region)
        ref_e = self.referrer_emb(referrer)
        
        # 时间embeddings
        m_e = self.month_emb(month)
        d_e = self.day_emb(day)
        h_e = self.hour_emb(hour)
        min_e = self.minute_emb(minute)
        
        # 历史序列pooling（位置递减权重）
        hist_embs = self.item_id_emb(history_ids)  # (B, K, D)
        K = history_ids.size(1)
        
        # 位置权重（指数衰减）
        beta = 0.5
        pos_idx = torch.arange(K, dtype=torch.float32, device=history_ids.device)
        weights = torch.exp(-beta * ((K - 1) - pos_idx))
        weights = weights / (weights.sum() + 1e-9)
        weights = weights.view(1, K, 1)
        
        hist_pool = (hist_embs * weights).sum(dim=1)  # (B, D)
        
        # 历史数值特征投影
        hist_num_proj = self.hist_num_proj(hist_num_feats)  # (B, 16)
        
        # 用户统计特征投影
        total_clicks_log = torch.log1p(user_stats[:, 0:1])
        active_days = user_stats[:, 1:2]
        stat_in = torch.cat([total_clicks_log, active_days], dim=1)
        stat_proj = self.user_stat_proj(stat_in)  # (B, 16)
        
        # 最后一次点击特征
        last_cat_e = self.category_emb(last_cat)
        last_cont_p = self.content_proj(last_content)  # (B, CONTENT_PROJ_DIM//2)
        
        # 拼接所有用户特征
        x = torch.cat([
            u_e, env_e, dev_e, os_e, country_e, region_e, ref_e,
            m_e, d_e, h_e, min_e, hist_pool, hist_num_proj, stat_proj,
            last_cat_e, last_cont_p
        ], dim=-1)
        
        user_vec = self.user_tower(x)  # (B, TOWER_OUTPUT_DIM)
        user_vec = F.normalize(user_vec, p=2, dim=1)
        
        return user_vec
    
    def forward_item_features(self, item_idxs, category_idxs, num_feats, content_feats):
        """
        Item Tower前向传播
        参数:
            item_idxs: (B,) 物品ID
            category_idxs: (B,) 类别ID
            num_feats: (B, 2) 数值特征（创建时间、词数）
            content_feats: (B, content_dim) 内容embedding
        返回:
            item_vec: (B, TOWER_OUTPUT_DIM) L2归一化
        """
        # 基础embeddings
        item_emb = self.item_id_emb(item_idxs)
        cat_e = self.category_emb(category_idxs)
        
        # 数值特征投影
        num_p = self.item_num_proj(num_feats)  # (B, CONTENT_PROJ_DIM//2)
        cont_p = self.content_proj(content_feats)  # (B, CONTENT_PROJ_DIM//2)
        
        # 拼接所有物品特征
        x = torch.cat([item_emb, cat_e, num_p, cont_p], dim=-1)
        
        item_vec = self.item_tower(x)  # (B, TOWER_OUTPUT_DIM)
        item_vec = F.normalize(item_vec, p=2, dim=1)
        
        return item_vec

def compute_scores(user_vecs, item_vecs, item_probs, alpha=0.5):
    """
    计算最终得分：cos_sim - alpha * log(p_i)
    参数:
        user_vecs: (B, D) 用户向量
        item_vecs: (B, C, D) 候选物品向量
        item_probs: (B, C) 物品采样概率
        alpha: 平衡超参数
    返回:
        scores: (B, C)
    """
    # 扩展用户向量
    B, C, D = item_vecs.shape
    user_vecs_exp = user_vecs.unsqueeze(1).expand(-1, C, -1)  # (B, C, D)
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(user_vecs_exp, item_vecs, dim=-1)  # (B, C)
    
    # 计算校正项
    log_p = torch.log(item_probs + 1e-12)
    
    # 最终得分
    scores = cos_sim - alpha * log_p
    
    return scores

if __name__ == "__main__":
    # 测试模型
    print("测试双塔模型...")
    
    # 模拟参数
    num_users = 1000
    num_items = 10000
    num_categories = 100
    max_env = 10
    max_device = 10
    max_os = 10
    max_country = 50
    max_region = 100
    max_referrer = 20
    
    # 创建模型
    model = TwinTowerFull(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
        max_env=max_env,
        max_device=max_device,
        max_os=max_os,
        max_country=max_country,
        max_region=max_region,
        max_referrer=max_referrer
    )
    
    # 测试输入
    batch_size = 32
    history_k = 20
    
    user_id = torch.randint(0, num_users, (batch_size,))
    env = torch.randint(0, max_env, (batch_size,))
    device_group = torch.randint(0, max_device, (batch_size,))
    os_ = torch.randint(0, max_os, (batch_size,))
    country = torch.randint(0, max_country, (batch_size,))
    region = torch.randint(0, max_region, (batch_size,))
    referrer = torch.randint(0, max_referrer, (batch_size,))
    month = torch.randint(1, 13, (batch_size,))
    day = torch.randint(1, 32, (batch_size,))
    hour = torch.randint(0, 24, (batch_size,))
    minute = torch.randint(0, 60, (batch_size,))
    history_ids = torch.randint(0, num_items, (batch_size, history_k))
    hist_num_feats = torch.randn(batch_size, 2)
    user_stats = torch.randn(batch_size, 2)
    last_cat = torch.randint(0, num_categories, (batch_size,))
    last_content = torch.randn(batch_size, 250)
    
    # 测试User Tower
    user_vec = model.forward_user(
        user_id, env, device_group, os_, country, region, referrer,
        month, day, hour, minute, history_ids, hist_num_feats, user_stats,
        last_cat, last_content
    )
    print(f"User向量形状: {user_vec.shape}")
    
    # 测试Item Tower
    item_idxs = torch.randint(0, num_items, (batch_size,))
    category_idxs = torch.randint(0, num_categories, (batch_size,))
    num_feats = torch.randn(batch_size, 2)
    content_feats = torch.randn(batch_size, 250)
    
    item_vec = model.forward_item_features(item_idxs, category_idxs, num_feats, content_feats)
    print(f"Item向量形状: {item_vec.shape}")
    
    # 测试得分计算
    item_vecs = item_vec.unsqueeze(1)  # (B, 1, D)
    item_probs = torch.ones(batch_size, 1) * 0.1  # (B, 1)
    
    scores = compute_scores(user_vec, item_vecs, item_probs)
    print(f"得分形状: {scores.shape}")
    
    print("模型测试完成！")