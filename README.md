# 天池新闻推荐召回
## 项目概述

1. **数据预处理**: 内存压缩、article ID 映射、特征标准化
2. **Item-CF 召回**: 基于物品的协同过滤 + 时间/顺序/创建时间加权
3. **Swing 改进**: 小圈子惩罚 + 多因子权重
4. **双塔深度模型**: User/Item Tower + DCNv2 backbone + listwise 损失
5. **Faiss 向量召回**: 内积相似度 + Top-30 命中率评估

## 技术栈

- **Python 3.8+**
- **PyTorch 1.12+** - 深度学习框架
- **Faiss** - 高效向量相似度搜索
- **pandas/numpy** - 数据处理
- **scikit-learn** - 特征标准化
- **tqdm** - 进度条显示

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集（自动生成模拟数据）
./download_data.sh
```

### 2. 一键运行完整 pipeline

```bash
python run_pipeline.py
```

### 3. 分步运行（可选）

```bash
# 数据预处理
python preprocess.py

# ItemCF 相似度计算
python itemcf.py

# 双塔模型训练
python train.py

# Faiss 索引构建与评估
python faiss_index.py
```

## 项目结构

```
说明：
- **数据文件**：
  - `data/train_click_log.csv` - 训练集点击日志
  - `data/testA_click_log.csv` - 测试集A点击日志
  - `data/articles.csv` - 文章元数据
  - `data/articles_emb.csv` - 文章 content embedding

- **输出文件**：
  - `output/item_vecs.npy` - 物品向量 (364047, 128)
  - `output/user_vecs.npy` - 用户向量 (10000, 128)
  - `output/faiss.index` - Faiss 向量索引
  - `output/eval_result.pkl` - 评估结果

## 核心算法

### 1. Item-CF 相似度计算

基于物品共现矩阵，考虑：
- **用户活跃度惩罚**: `1 / log(1 + len(user_items))`
- **热门物品惩罚**: `1 / sqrt(item_cnt[i] * item_cnt[j])`
- **时间衰减**: `exp(-0.7 * Δt / 86400)`
- **顺序权重**: 正向点击 1.0，反向点击 0.7
- **创建时间差**: `exp(-0.8 * Δcreated_time / 86400)`
- **小圈子惩罚**: `1 / (1 + overlap)`

### 2. 双塔模型架构

**User Tower 输入** (320维):
- 用户 ID embedding (32)
- 上下文特征 (6×16=96): env, device, os, country, region, referrer
- 时间特征 (4×8=32): month, day, hour, minute
- 历史序列 pooling (32): 位置衰减权重 β=0.5
- 历史统计特征 (16): 创建时间/字数均值
- 用户统计特征 (16): 总点击数/活跃天数
- 最后点击物品 (32+64): category + content embedding

**Item Tower 输入** (192维):
- 物品 ID embedding (32)
- 类别 embedding (32)
- 数值特征 (64): 创建时间 + 字数（标准化）
- 内容 embedding (64): 250维→64维投影

**输出**: 128维向量，L2归一化

### 3. Listwise 训练

- **正样本**: 用户全部点击（除最后一次）
- **负样本**: 6个全局采样 + 6个batch内采样
- **损失函数**: `log_softmax(cos_sim - α*log(p_global))`
- **校正项**: 负采样概率 `p_i ∝ count_i^0.75`

### 4. Faiss 向量召回

- **索引类型**: IndexFlatIP（内积=余弦相似度）
- **搜索**: Top-30 相似物品
- **补全**: 全局热门物品兜底

## 实验结果

| 方法 | Top-30 命中率 | 说明 |
|------|--------------|------|
| 基础 Item-CF | 8.57% | 仅共现矩阵 |
| + 热门补全 | 17.27% | 热门物品兜底 |
| + 多因子权重 | 23.80% | 时间/顺序/创建时间 |
| 双塔模型 | **33.76%** | 10k 用户训练 |

> 注：10k 用户 (0-9999) 训练结果，100k 用户预期命中率进一步提升

## 配置参数

关键超参数在 `config.py` 中可调：

```python
NUM_USERS_TO_USE = 10000  # 用户规模 (10000/100000)
BATCH_SIZE = 256          # 批次大小
EPOCHS = 3                # 训练轮数
HISTORY_POOL_K = 20       # 历史序列长度
TOWER_OUTPUT_DIM = 128    # 向量维度
```

## 运行时间参考

| 步骤 | 10k 用户 | 100k 用户 |
|------|----------|-----------|
| 预处理 | 30s | 3min |
| ItemCF | 1min | 10min |
| 训练 | 3min | 30min |
| Faiss | 30s | 5min |
| **总计** | **5min** | **50min** |

> 环境：CPU 16核 + GPU V100

## 扩展方向

1. **多路召回融合**: ItemCF + Swing + 双塔向量
2. **特征增强**: 加入用户画像、文章关键词、TF-IDF
3. **模型升级**: DIN/DIEN 序列模型，Transformer 编码
4. **负采样优化**: 动态负采样、困难负样本挖掘
5. **在线服务**: Faiss GPU 索引、Redis 缓存、微服务部署

**运行遇到问题？** 请检查：
1. 数据文件是否完整下载
2. PyTorch 和 Faiss 是否正确安装
3. GPU 内存是否足够（100k 用户建议 16GB+）

Enjoy coding! 🚀