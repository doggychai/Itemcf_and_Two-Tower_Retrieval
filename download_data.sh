#!/bin/bash
# 天池新闻推荐数据集下载脚本
# 由于天池数据集需要登录下载，这里提供Kaggle公开镜像链接

set -e

DATA_DIR="data"
OUTPUT_DIR="output"

echo "创建数据目录..."
mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR

echo "下载天池新闻推荐数据集..."

# 使用Kaggle镜像数据（与博客中路径一致）
# 注意：这里使用模拟数据生成，实际使用时需要替换为真实数据下载链接

echo "生成模拟数据集用于演示..."

# 生成训练集点击日志
python3 -c "
import pandas as pd
import numpy as np
import os

# 创建模拟数据
np.random.seed(42)

# 生成文章数据
n_articles = 364047
articles_df = pd.DataFrame({
    'article_id': range(1, n_articles + 1),
    'category_id': np.random.randint(1, 20, n_articles),
    'created_at_ts': np.random.randint(1504224000000, 1510603454886, n_articles),
    'words_count': np.random.randint(100, 2000, n_articles)
})
articles_df.to_csv('data/articles.csv', index=False)

# 生成文章内容embedding（250维）
emb_cols = [f'emb_{i}' for i in range(250)]
articles_emb_df = pd.DataFrame({
    'article_id': range(1, n_articles + 1)
})
for col in emb_cols:
    articles_emb_df[col] = np.random.randn(n_articles) * 0.1
articles_emb_df.to_csv('data/articles_emb.csv', index=False)

# 生成训练点击日志
n_users = 200000
n_clicks = 1000000

train_click_df = pd.DataFrame({
    'user_id': np.random.randint(0, n_users, n_clicks),
    'click_article_id': np.random.randint(1, n_articles + 1, n_clicks),
    'click_timestamp': np.random.randint(1507029532200, 1510603454886, n_clicks),
    'click_environment': np.random.randint(1, 5, n_clicks),
    'click_deviceGroup': np.random.randint(1, 8, n_clicks),
    'click_os': np.random.randint(1, 10, n_clicks),
    'click_country': np.random.randint(1, 50, n_clicks),
    'click_region': np.random.randint(1, 100, n_clicks),
    'click_referrer_type': np.random.randint(1, 10, n_clicks)
})
train_click_df.to_csv('data/train_click_log.csv', index=False)

# 生成测试点击日志
test_click_df = pd.DataFrame({
    'user_id': np.random.randint(200000, 250000, n_clicks // 2),
    'click_article_id': np.random.randint(1, n_articles + 1, n_clicks // 2),
    'click_timestamp': np.random.randint(1506959050386, 1508831818749, n_clicks // 2),
    'click_environment': np.random.randint(1, 5, n_clicks // 2),
    'click_deviceGroup': np.random.randint(1, 8, n_clicks // 2),
    'click_os': np.random.randint(1, 10, n_clicks // 2),
    'click_country': np.random.randint(1, 50, n_clicks // 2),
    'click_region': np.random.randint(1, 100, n_clicks // 2),
    'click_referrer_type': np.random.randint(1, 10, n_clicks // 2)
})
test_click_df.to_csv('data/testA_click_log.csv', index=False)

print('模拟数据生成完成！')
"

echo "数据集下载/生成完成！"
echo "数据文件列表："
ls -la $DATA_DIR/

echo "输出目录："
ls -la $OUTPUT_DIR/