"""
天池新闻推荐 - 数据集构建模块
NewsListwiseFullDataset + collate_fn
支持全局负采样 + batch内负采样
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd

from config import *

class NewsListwiseFullDataset(Dataset):
    """
    双塔训练数据集
    """
    def __init__(self, sample_idx_list, user_clicks, user_stats, item_probs, history_k=HISTORY_POOL_K):
        self.samples = sample_idx_list
        self.user_clicks = user_clicks
        self.user_stats = user_stats
        self.item_probs = item_probs
        self.history_k = history_k
        self.all_items = np.arange(1, len(item_probs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 注意：这里解包必须和 train.py 中 sample_idx_list 的格式一致
        u, pos_idx = self.samples[idx]
        clicks = self.user_clicks[u]
        pos_rec = clicks[pos_idx]

        # 正样本物品
        pos_item = pos_rec['item_idx']
        pos_ts = pos_rec['timestamp']

        # 时间特征
        dt = pd.to_datetime(pos_ts, unit='ms')
        month = int(dt.month)
        day = int(dt.day)
        hour = int(dt.hour)
        minute = int(dt.minute)

        # 上下文特征
        env = int(pos_rec['env'])
        device_group = int(pos_rec['device_group'])
        os_ = int(pos_rec['os'])
        country = int(pos_rec['country'])
        region = int(pos_rec['region'])
        referrer = int(pos_rec['referrer'])

        # 历史序列（不包含当前点击）
        # pos_idx 是当前点击在序列中的索引，历史取 pos_idx 之前的
        start = max(0, pos_idx - self.history_k)
        hist_items = [clicks[i]['item_idx'] for i in range(start, pos_idx)]

        # 补齐历史序列
        if len(hist_items) < self.history_k:
            pad_len = self.history_k - len(hist_items)
            hist_items = ([0] * pad_len) + hist_items
        else:
            hist_items = hist_items[-self.history_k:]

        # 全局负采样
        global_negs = list(np.random.choice(
            self.all_items,
            size=GLOBAL_NEG_PER_SAMPLE,
            replace=False,
            p=self.item_probs
        ))

        # 用户统计特征 (KeyError 修复依赖 train.py 的 user_stats 构建)
        total_clicks = self.user_stats[u]['total_clicks']
        active_days = self.user_stats[u]['active_days']
        last_item = self.user_stats[u]['last_item']

        return {
            'user_id': int(u),
            'env': env,
            'device_group': device_group,
            'os': os_,
            'country': country,
            'region': region,
            'referrer': referrer,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'history': np.array(hist_items, dtype=np.int64),
            'pos_item': int(pos_item),
            'global_negs': np.array(global_negs, dtype=np.int64),
            'pos_ts': int(pos_ts),
            'total_clicks': int(total_clicks),
            'active_days': int(active_days),
            'last_item': int(last_item)
        }

class Collator:
    """
    使用类来封装 collate_fn，以便传入 item_probs
    """
    def __init__(self, item_probs):
        self.item_probs = item_probs
        self.num_items = len(item_probs)

    def __call__(self, batch):
        bs = len(batch)

        # 使用numpy数组避免tensor警告
        history = np.array([b['history'] for b in batch], dtype=np.int64)

        # 基本特征
        user_id = torch.LongTensor([b['user_id'] for b in batch])
        env = torch.LongTensor([b['env'] for b in batch])
        device_group = torch.LongTensor([b['device_group'] for b in batch])
        os_ = torch.LongTensor([b['os'] for b in batch])
        country = torch.LongTensor([b['country'] for b in batch])
        region = torch.LongTensor([b['region'] for b in batch])
        referrer = torch.LongTensor([b['referrer'] for b in batch])
        month = torch.LongTensor([b['month'] for b in batch])
        day = torch.LongTensor([b['day'] for b in batch])
        hour = torch.LongTensor([b['hour'] for b in batch])
        minute = torch.LongTensor([b['minute'] for b in batch])

        # 历史序列
        history_t = torch.from_numpy(history).long()

        # 正样本
        pos_items = [int(b['pos_item']) for b in batch]
        pos_ts = torch.LongTensor([b['pos_ts'] for b in batch])

        # 用户统计
        total_clicks = torch.FloatTensor([b['total_clicks'] for b in batch])
        active_days = torch.FloatTensor([b['active_days'] for b in batch])
        last_items = [int(b['last_item']) for b in batch]

        # 全局负样本
        global_negs = np.stack([b['global_negs'] for b in batch], axis=0)

        # batch内负采样
        pool_pos = pos_items.copy()
        batch_negs = np.zeros((bs, BATCH_NEG_PER_SAMPLE), dtype=np.int64)

        for i in range(bs):
            # 从其他样本的正样本中采样
            candidates = [p for j, p in enumerate(pool_pos) if j != i]

            if len(candidates) >= BATCH_NEG_PER_SAMPLE:
                chosen = random.sample(candidates, BATCH_NEG_PER_SAMPLE)
            else:
                chosen = candidates.copy()
                need = BATCH_NEG_PER_SAMPLE - len(chosen)

                # 从全局负样本中补充 (flat_global)
                flat_global = [int(x) for r in global_negs for x in r
                            if int(x) not in chosen and int(x) != pos_items[i]]

                if len(flat_global) >= need:
                    chosen += random.sample(flat_global, need)
                else:
                    # 随机采样 (这里需要 self.item_probs)
                    extra = list(np.random.choice(
                        np.arange(self.num_items),
                        size=need,
                        p=self.item_probs
                    ))
                    chosen += [int(x) for x in extra]

            batch_negs[i, :] = np.array(chosen, dtype=np.int64)

        # 构建候选矩阵：1正 + 6全局负 + 6batch负 = 13
        candidates = np.concatenate([
            np.array(pos_items)[:, None],
            global_negs,
            batch_negs
        ], axis=1)

        candidates_t = torch.from_numpy(candidates).long()

        return {
            'user_id': user_id,
            'env': env,
            'device_group': device_group,
            'os': os_,
            'country': country,
            'region': region,
            'referrer': referrer,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'history': history_t,
            'pos_items': torch.LongTensor(pos_items),
            'candidates': candidates_t,
            'pos_ts': pos_ts,
            'total_clicks': total_clicks,
            'active_days': active_days,
            'last_items': torch.LongTensor(last_items)
        }

def create_dataloaders(sample_idx_list, user_clicks, user_stats, item_probs):
    """
    创建训练集DataLoader
    """
    dataset = NewsListwiseFullDataset(
        sample_idx_list,
        user_clicks,
        user_stats,
        item_probs,
        history_k=HISTORY_POOL_K
    )

    # 初始化 Collator
    collator = Collator(item_probs)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator, # 使用实例作为 collate_fn
        num_workers=2,
        pin_memory=True
    )

    return dataloader

if __name__ == "__main__":
    pass