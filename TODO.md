# 天池新闻推荐召回

**技术栈：** Python 3.8 + pandas + numpy + scikit-learn + PyTorch 1.12 + Faiss + tqdm

---

## 任务清单

### Task 1: 项目骨架与数据下载

**涉及文件：**
- 创建: `requirements.txt`（依赖列表）
- 创建: `download_data.sh`（自动下载天池数据集脚本）
- 创建: `config.py`（全局超参与路径配置，支持 10 k / 100 k 切换）

**步骤：**
- [ ] 1.1 写入依赖：torch、faiss-cpu、sklearn、tqdm、pandas、numpy
- - [ ] 1.2 脚本拉取 `train_click_log.csv`、`testA_click_log.csv`、`articles.csv`、`articles_emb.csv` 到 `data/` 目录
- [ ] 1.3 配置文件中定义 `NUM_USERS_TO_USE`、`BATCH_SIZE`、`EPOCHS`、`HISTORY_POOL_K` 等开关

### Task 2: 数据预处理与特征工程

**涉及文件：**
- 创建: `preprocess.py`（内存压缩、article→连续 idx、时间/数值标准化、用户点击序列构建）
- 创建: `features.py`（生成 item 创建时间字典、负采样权重 p_global、用户统计量）

**步骤：**
- [ ] 2.1 实现 `reduce_mem()` 复现内存压缩函数
- [ ] 2.2 构建 `article_id_to_idx` 映射，生成 `category_arr`、`created_arr`、`words_arr`、`content_emb_arr`
- [ ] 2.3 按用户聚合点击序列并排序，输出 `user_clicks.pkl` 与 `item_click_count.pkl`
- [ ] 2.4 计算全局负采样概率 `item_sampling_probs.npy` ∝ count^0.75

### Task 3: Item-CF / Swing 相似度矩阵

**涉及文件：**
- 创建: `itemcf.py`（Item-CF 共现权重 + 时间/顺序/创建时间差加权）
- 创建: `swing.py`（Swing 小圈子惩罚实现，可调 α）

**步骤：**
- [ ] 3.1 实现 `build_user_item_time()` 与 `itemcf_sim()`，保存 `itemcf_i2i_sim.pkl`
- [ ] 3.2 在 Swing 分支加入位置权重、点击时间衰减、文章创建时间衰减，输出 `item_sim_enhanced.pkl`
- [ ] 3.3 构建 `item_topk.pkl`（每篇文章 top-20 相似列表）

### Task 4: 双塔深度模型训练

**涉及文件：**
- 创建: `model.py`（TwinTowerFull + DCNv2Block）
- 创建: `dataset.py`（NewsListwiseFullDataset + collate_fn）
- 创建: `train.py`（训练循环、listwise 损失、Faiss 向量导出）

**步骤：**
- [ ] 4.1 搭建 User/Item Tower，维度严格对齐博客（320/192 → 128）
- [ ] 4.2 Dataset 支持“全部点击（除最后一次）”正样本，历史 20 截断 + 位置衰减权重
- [ ] 4.3 collate 内完成 global & batch 负采样，候选列 13（1+6+6）
- [ ] 4.4 训练脚本支持多 epoch、Adam、log-softmax 损失，实时打印命中率
- [ ] 4.5 训练结束导出 `item_vecs.npy` 与 `user_vecs.npy`（100 k 用户可分批推理）

### Task 5: Faiss 向量召回与评估

**涉及文件：**
- 创建: `faiss_index.py`（IndexFlatIP 构建、L2 归一化）
- 创建: `evaluate.py`（top-30 召回、命中率、补全策略）

**步骤：**
- [ ] 5.1 加载 item 向量，归一化后 add 到 Faiss 索引
- [ ] 5.2 批量推理 100 k 用户向量，search top-30，输出 `recall_top30.csv`
- [ ] 5.3 计算命中率（hit@30）与平均补全数量，打印日志
- [ ] 5.4 支持热门补全兜底（global top-100 热门文章）

### Task 6: 入口脚本与一键运行

**涉及文件：**
- 创建: `run_full_pipeline.sh`（顺序执行 preprocess→itemcf→train→eval）
- 创建: `README.md`（运行命令、结果指标、参数说明）

**步骤：**
- [ ] 6.1 串行调用各模块，支持 10 k 快速验证与 100 k 全量模式
- [ ] 6.2 README 给出各阶段输出文件与预期指标（hit@30 ≈ 0.34 for 10 k）
- [ ] 6.3 提供 tensorboard 日志目录（可选）

### Task 7: 项目验证与完成

**步骤：**
- [ ] 7.1 运行 `run_full_pipeline.sh` 无报错，控制台无警告
- [ ] 7.2 检查输出：item_sim 矩阵、item_vecs、recall_top30、命中率日志
- [ ] 7.3 100 k 用户模式下内存 < 32 GB、训练时间 < 2 h（GPU）
- [ ] 7.4 打包 `output/` 目录，包含模型权重与评估报告