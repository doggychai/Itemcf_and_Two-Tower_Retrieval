"""
天池新闻推荐 baseline 配置文件
支持 10k/100k 用户规模一键切换
"""

import os
import torch

# =====================
# 路径配置
# =====================
DATA_DIR = "data"
OUTPUT_DIR = "output"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = os.path.join(DATA_DIR, "train_click_log.csv")
TEST_PATH = os.path.join(DATA_DIR, "testA_click_log.csv")
ARTICLES_PATH = os.path.join(DATA_DIR, "articles.csv")
ARTICLES_EMB_PATH = os.path.join(DATA_DIR, "articles_emb.csv")

# =====================
# 数据规模开关
# =====================
# 10k 快速验证模式 (调试用)
# NUM_USERS_TO_USE = 10000
# 100k 全量模式 (正式跑)
NUM_USERS_TO_USE = 10000  # 如果你想跑全量，可以设为 None 或者更大的数字

# =====================
# 训练超参
# =====================
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 特征维度
# =====================
ITEM_ID_EMB_DIM = 32
USER_ID_EMB_DIM = 32
CAT_EMB_DIM = 32
OTHER_CAT_EMB_DIM = 16
TIME_EMB_DIM = 8
CONTENT_PROJ_DIM = 128
TOWER_OUTPUT_DIM = 128

# =====================
# 负采样
# =====================
GLOBAL_NEG_PER_SAMPLE = 6
BATCH_NEG_PER_SAMPLE = 6
NEG_NUM = GLOBAL_NEG_PER_SAMPLE + BATCH_NEG_PER_SAMPLE

# =====================
# 历史序列
# =====================
HISTORY_POOL_K = 20

# =====================
# 召回评估
# =====================
TOPK = 30

# =====================
# 随机种子
# =====================
SEED = 42

# =====================
# 中间文件 (Output Files)
# =====================

# 1. 预处理相关
ARTICLE_MAPPING_PKL = os.path.join(OUTPUT_DIR, 'article_mapping.pkl')
CATEGORY_ARR_NPY = os.path.join(OUTPUT_DIR, 'category_arr.npy')
CREATED_ARR_NPY = os.path.join(OUTPUT_DIR, 'created_arr.npy')
WORDS_ARR_NPY = os.path.join(OUTPUT_DIR, 'words_arr.npy')
CONTENT_EMB_ARR_NPY = os.path.join(OUTPUT_DIR, 'content_emb_arr.npy')
USER_CLICKS_PKL = os.path.join(OUTPUT_DIR, "user_clicks.pkl")
ITEM_CLICK_COUNT_PKL = os.path.join(OUTPUT_DIR, "item_click_count.pkl")
ITEM_SAMPLING_PROBS_NPY = os.path.join(OUTPUT_DIR, 'item_sampling_probs.npy')

# 2. 召回模型/向量相关 (DSSM等)
ITEM_VECS_NPY = os.path.join(OUTPUT_DIR, "item_vecs.npy")
USER_VECS_NPY = os.path.join(OUTPUT_DIR, "user_vecs.npy")
RECALl_TOP30_CSV = os.path.join(OUTPUT_DIR, "recall_top30.csv")

# 3. 传统召回相关 (ItemCF & Swing) - 【这里是你缺少的部分】
# 通用的变量（如果你只想用一个文件存）
ITEM_SIM_PKL = os.path.join(OUTPUT_DIR, "item_sim_enhanced.pkl")
ITEM_TOPK_PKL = os.path.join(OUTPUT_DIR, "item_topk.pkl")

# 专门针对 ItemCF 算法的输出
ITEMCF_SIM_PKL = os.path.join(OUTPUT_DIR, "itemcf_sim.pkl")
ITEMCF_TOPK_PKL = os.path.join(OUTPUT_DIR, "itemcf_topk.pkl")

# 专门针对 Swing 算法的输出
SWING_SIM_PKL = os.path.join(OUTPUT_DIR, "swing_sim.pkl")
SWING_TOPK_PKL = os.path.join(OUTPUT_DIR, "swing_topk.pkl")

# 热门物品补全
TOPK_CLICK_PKL = os.path.join(OUTPUT_DIR, "topk_click.pkl")

# FAISS
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")

#模型保存
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pth")

EVAL_RESULT_PATH = os.path.join(OUTPUT_DIR, "eval_result.pkl")
