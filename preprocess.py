"""
天池新闻推荐 - 数据预处理模块
包含内存压缩、article映射、特征数组构建
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm

from config import *

def reduce_mem(df):
    """
    内存压缩函数 - 复现博客中的reduce_mem函数
    将数值型列转换为更小的数据类型以节省内存
    """
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction), time spend: {:2.2f} min'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem, (time.time() - starttime) / 60))
    return df

def load_and_filter_data():
    """
    加载数据并过滤用户
    返回: train_df, articles_df, articles_emb_df
    """
    print("读取CSV文件...")
    train_df = pd.read_csv(TRAIN_PATH)
    articles_df = pd.read_csv(ARTICLES_PATH)
    articles_emb_df = pd.read_csv(ARTICLES_EMB_PATH)
    
    print("原始训练集大小:", len(train_df))
    print("原始文章数:", len(articles_df))
    
    # 内存压缩
    train_df = reduce_mem(train_df)
    articles_df = reduce_mem(articles_df)
    articles_emb_df = reduce_mem(articles_emb_df)
    
    # 过滤用户
    print(f"筛选前{NUM_USERS_TO_USE}个用户...")
    valid_users = set([u for u in train_df['user_id'].unique() if u <= (NUM_USERS_TO_USE - 1)])
    train_df = train_df[train_df['user_id'].isin(valid_users)].copy()
    train_df.reset_index(drop=True, inplace=True)
    
    print("过滤后训练点击条数:", len(train_df))
    print("参与用户数:", len(valid_users))
    
    return train_df, articles_df, articles_emb_df

def build_article_mappings(articles_df, articles_emb_df):
    """
    构建article_id到连续idx的映射
    返回: article_id_to_idx, idx_to_article_id, NUM_ITEMS
    """
    unique_articles = articles_df['article_id'].unique().tolist()
    article_id_to_idx = {aid: idx+1 for idx, aid in enumerate(unique_articles)}  # 1-based, 0 for padding
    idx_to_article_id = {v: k for k, v in article_id_to_idx.items()}
    NUM_ITEMS = len(article_id_to_idx) + 1  # include padding 0
    
    print("文章数:", NUM_ITEMS-1)
    
    # 保存映射
    with open(ARTICLE_MAPPING_PKL, 'wb') as f:
        pickle.dump({'article_id_to_idx': article_id_to_idx, 
                    'idx_to_article_id': idx_to_article_id}, f)
    
    return article_id_to_idx, idx_to_article_id, NUM_ITEMS

def build_article_features(articles_df, articles_emb_df, article_id_to_idx):
    """
    构建article特征数组（与idx对齐）
    返回: category_arr, created_arr, words_arr, content_emb_arr
    """
    content_cols = [c for c in articles_emb_df.columns if c.startswith("emb_")]
    content_dim = len(content_cols)
    print("content dim:", content_dim)
    
    NUM_ITEMS = len(article_id_to_idx) + 1
    
    # 预分配数组（index 0 reserved for padding）
    category_arr = np.zeros((NUM_ITEMS,), dtype=np.int64)
    created_arr = np.zeros((NUM_ITEMS,), dtype=np.float32)
    words_arr = np.zeros((NUM_ITEMS,), dtype=np.float32)
    content_emb_arr = np.zeros((NUM_ITEMS, content_dim), dtype=np.float32)
    
    # 临时列表用于标准化
    tmp_created = []
    tmp_words = []
    tmp_index_for_scaler = []
    
    # 填充基础特征
    for row in articles_df.itertuples(index=False):
        aid = int(row.article_id)
        if aid not in article_id_to_idx:
            continue
        idx = article_id_to_idx[aid]
        
        category_arr[idx] = int(row.category_id)
        created_arr[idx] = float(row.created_at_ts)
        words_arr[idx] = float(row.words_count)
        
        tmp_created.append(created_arr[idx])
        tmp_words.append(words_arr[idx])
        tmp_index_for_scaler.append(idx)
    
    # 标准化数值特征
    if len(tmp_index_for_scaler) > 0:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.stack([tmp_created, tmp_words], axis=1))
        for k, idx in enumerate(tmp_index_for_scaler):
            created_arr[idx] = float(scaled[k, 0])
            words_arr[idx] = float(scaled[k, 1])
    
    # 填充content embedding
    for row in articles_emb_df.itertuples(index=False):
        aid = int(row.article_id)
        if aid not in article_id_to_idx:
            continue
        idx = article_id_to_idx[aid]
        content_emb_arr[idx] = np.array(row[1:], dtype=np.float32)
    
    print("article features prepared")
    
    # 保存特征数组
    np.save(CATEGORY_ARR_NPY, category_arr)
    np.save(CREATED_ARR_NPY, created_arr)
    np.save(WORDS_ARR_NPY, words_arr)
    np.save(CONTENT_EMB_ARR_NPY, content_emb_arr)
    
    return category_arr, created_arr, words_arr, content_emb_arr

def build_user_clicks(train_df, article_id_to_idx):
    """
    构建用户点击序列（按时间排序）
    返回: user_clicks, item_click_count
    """
    user_clicks = defaultdict(list)
    item_click_count = defaultdict(int)
    
    print("构建user clicks...")
    for r in tqdm(train_df.itertuples(index=False), total=len(train_df)):
        uid = int(r.user_id)
        aid = int(r.click_article_id)
        
        if aid not in article_id_to_idx:
            continue
        idx = article_id_to_idx[aid]
        
        ts = int(r.click_timestamp)
        env = int(r.click_environment)
        device_group = int(r.click_deviceGroup)
        os_ = int(r.click_os)
        country = int(r.click_country)
        region = int(r.click_region)
        referrer = int(r.click_referrer_type)
        
        # 添加点击记录
        user_clicks[uid].append({
            'item_idx': idx,
            'timestamp': ts,
            'env': env,
            'device_group': device_group,
            'os': os_,
            'country': country,
            'region': region,
            'referrer': referrer
        })
        
        item_click_count[idx] += 1
    
    # 按时间排序
    for u in list(user_clicks.keys()):
        user_clicks[u].sort(key=lambda x: x['timestamp'])
    
    print("构建user clicks完成，用户数:", len(user_clicks))
    
    # 保存
    with open(USER_CLICKS_PKL, 'wb') as f:
        pickle.dump(dict(user_clicks), f)
    
    with open(ITEM_CLICK_COUNT_PKL, 'wb') as f:
        pickle.dump(dict(item_click_count), f)
    
    return dict(user_clicks), dict(item_click_count)

def build_sampling_probs(item_click_count, NUM_ITEMS):
    """
    构建全局负采样概率 p ~ cnt^0.75
    返回: item_sampling_probs (numpy array)
    """
    weights = np.zeros((NUM_ITEMS,), dtype=np.float64)
    
    for idx in range(1, NUM_ITEMS):  # skip padding 0
        cnt = item_click_count.get(idx, 0)
        if cnt > 0:
            weights[idx] = cnt ** 0.75
    
    if weights.sum() == 0:
        weights += 1.0
    
    weights = weights / weights.sum()
    
    # 保存
    np.save(ITEM_SAMPLING_PROBS_NPY, weights)
    
    print("全局负采样概率构建完成")
    return weights


def load_preprocessed_data():
    """
    加载预处理后的所有数据（完整版本）
    """
    import pickle
    import numpy as np

    # ===== 1. mapping =====
    with open(ARTICLE_MAPPING_PKL, 'rb') as f:
        mapping = pickle.load(f)

    article_id_to_idx = mapping['article_id_to_idx']
    idx_to_article_id = mapping['idx_to_article_id']
    NUM_ITEMS = len(article_id_to_idx) + 1

    # ===== 2. 特征 =====
    category_arr = np.load(CATEGORY_ARR_NPY)
    created_arr = np.load(CREATED_ARR_NPY)
    words_arr = np.load(WORDS_ARR_NPY)
    content_emb_arr = np.load(CONTENT_EMB_ARR_NPY)

    # ===== 3. 用户点击 =====
    with open(USER_CLICKS_PKL, 'rb') as f:
        user_clicks = pickle.load(f)

    # ===== 4. item click count =====
    with open(ITEM_CLICK_COUNT_PKL, 'rb') as f:
        item_click_count = pickle.load(f)

    # ===== 5. 自动推断统计信息（关键补全！）=====

    # category
    num_categories = int(category_arr.max()) + 1

    # 用户统计（你代码需要）
    user_stats = {}
    for uid, clicks in user_clicks.items():
        timestamps = [c['timestamp'] for c in clicks]
        if len(timestamps) == 0:
            continue

        days = set([t // (1000 * 60 * 60 * 24) for t in timestamps])

        user_stats[uid] = {
            'total_clicks': len(clicks),
            'active_days': len(days)
        }

    # 环境类最大值
    max_env = 0
    max_device = 0
    max_os = 0
    max_country = 0
    max_region = 0
    max_referrer = 0

    for clicks in user_clicks.values():
        for c in clicks:
            max_env = max(max_env, c['env'])
            max_device = max(max_device, c['device_group'])
            max_os = max(max_os, c['os'])
            max_country = max(max_country, c['country'])
            max_region = max(max_region, c['region'])
            max_referrer = max(max_referrer, c['referrer'])

    return {
        # mapping
        "article_id_to_idx": article_id_to_idx,
        "idx_to_article_id": idx_to_article_id,
        "NUM_ITEMS": NUM_ITEMS,

        # 特征
        "category_arr": category_arr,
        "created_arr": created_arr,
        "words_arr": words_arr,
        "content_emb_arr": content_emb_arr,

        # 用户
        "user_clicks": user_clicks,
        "user_stats": user_stats,

        # 统计信息（给模型用）
        "num_categories": num_categories,
        "max_env": max_env + 1,
        "max_device": max_device + 1,
        "max_os": max_os + 1,
        "max_country": max_country + 1,
        "max_region": max_region + 1,
        "max_referrer": max_referrer + 1
    }


def main():
    """主函数：执行完整的数据预处理流程"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载并过滤数据
    train_df, articles_df, articles_emb_df = load_and_filter_data()
    
    # 2. 构建article映射
    article_id_to_idx, idx_to_article_id, NUM_ITEMS = build_article_mappings(articles_df, articles_emb_df)
    
    # 3. 构建article特征数组
    category_arr, created_arr, words_arr, content_emb_arr = build_article_features(
        articles_df, articles_emb_df, article_id_to_idx
    )
    
    # 4. 构建用户点击序列
    user_clicks, item_click_count = build_user_clicks(train_df, article_id_to_idx)
    
    # 5. 构建负采样概率
    item_sampling_probs = build_sampling_probs(item_click_count, NUM_ITEMS)
    
    print("数据预处理全部完成！")
    print("输出文件列表：")
    for file in [ARTICLE_MAPPING_PKL, CATEGORY_ARR_NPY, CREATED_ARR_NPY, 
                 WORDS_ARR_NPY, CONTENT_EMB_ARR_NPY, USER_CLICKS_PKL, 
                 ITEM_CLICK_COUNT_PKL, ITEM_SAMPLING_PROBS_NPY]:
        if os.path.exists(file):
            print(f"  {file}")

if __name__ == "__main__":
    main()