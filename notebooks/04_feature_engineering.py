
import pandas as pd
import numpy as np

# ========== 加载数据 ==========
df = pd.read_csv('../data/cleaned_nev_data.csv')
print(f"原始数据: {df.shape}")

# ========== 1. 时间特征 ==========
df['年月'] = df['年份'].astype(str) + '-' + df['月份'].astype(str).str.zfill(2)

# 是否春节月（1-2月）
df['is_spring_festival'] = df['月份'].isin([1, 2]).astype(int)

# 是否旺季（9-12月，年底冲量）
df['is_peak_season'] = df['月份'].isin([9, 10, 11, 12]).astype(int)

# ========== 2. 城市层级 ==========
tier1 = ['北京', '上海', '广州', '深圳']
tier2 = ['成都', '杭州', '武汉', '西安', '苏州', '天津', '南京', '重庆', '郑州', '长沙']

df['city_tier'] = 2  # 默认其他=2
df.loc[df['城市'].isin(tier1), 'city_tier'] = 0  # 一线=0
df.loc[df['城市'].isin(tier2), 'city_tier'] = 1  # 新一线=1

# ========== 3. 价格段 ==========
df['price_segment'] = pd.cut(
    df['price_avg'],
    bins=[0, 10, 15, 20, 30, 50, 100],
    labels=['0_10', '10_15', '15_20', '20_30', '30_50', '50plus']
).astype(str)

# ========== 4. 品牌历史销量（用2022-2025数据算，模拟"已知信息"） ==========
# 每个品牌到当前月份为止的累计销量（作为品牌热度的代理）
brand_cumsum = df.sort_values(['年份', '月份']).groupby('品牌')['上牌量'].cumsum()
df['brand_cumulative_sales'] = brand_cumsum

# 品牌排名（按总销量）
brand_rank = df.groupby('品牌')['上牌量'].sum().rank(ascending=False)
df['brand_rank'] = df['品牌'].map(brand_rank)

# ========== 5. 车型历史销量 ==========
model_cumsum = df.sort_values(['年份', '月份']).groupby('车型')['上牌量'].cumsum()
df['model_cumulative_sales'] = model_cumsum

# ========== 6. 省份销量分档 ==========
province_total = df.groupby('省份')['上牌量'].sum()
province_bins = pd.qcut(province_total, q=3, labels=['low', 'mid', 'high'])
df['province_tier'] = df['省份'].map(province_bins)

# ========== 7. 滞后特征（上个月该车型在该城市的销量） ==========
df = df.sort_values(['车型', '城市', '年份', '月份'])
df['last_month_sales'] = df.groupby(['车型', '城市'])['上牌量'].shift(1)
df['last_month_sales'] = df['last_month_sales'].fillna(0)

# ========== 8. 目标变量：下个月销量（用于预测） ==========
df['next_month_sales'] = df.groupby(['车型', '城市'])['上牌量'].shift(-1)

# ========== 保存特征工程后的数据 ==========
feature_cols = [
    '年份', '月份', '省份', '城市', '品牌', '车型',
    '上牌量', 'price_avg', '上牌量_capped',
    'is_spring_festival', 'is_peak_season', 'city_tier',
    'price_segment', 'brand_cumulative_sales', 'brand_rank',
    'model_cumulative_sales', 'province_tier',
    'last_month_sales', 'next_month_sales'
]

df_features = df[feature_cols]
df_features.to_csv('../data/features_with_target.csv', index=False, encoding='utf-8-sig')
print(f"\n特征工程完成，输出数据: {df_features.shape}")
print(f"特征列: {[c for c in df_features.columns if c not in ['上牌量', 'next_month_sales']]}")
print("\n前5行预览:")
print(df_features.head())
print("\n=== 特征工程完成 ===")