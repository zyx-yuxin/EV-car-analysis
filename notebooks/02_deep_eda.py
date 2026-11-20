
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载数据 ==========
df = pd.read_csv('../data/cleaned_nev_data.csv')
print(f"数据加载完成: {df.shape}")

# ========== 方向一：品牌份额随时间变化 ==========
print("\n--- 方向一：品牌份额变化 ---")
year_brand = df.groupby(['年份', '品牌'])['上牌量'].sum().reset_index()

# 修复：给 year_total 改名，避免 merge 时列名冲突
year_total = df.groupby('年份')['上牌量'].sum().rename('年总量')
year_brand = year_brand.merge(year_total, on='年份')
year_brand['份额'] = year_brand['上牌量'] / year_brand['年总量'] * 100

top8 = df.groupby('品牌')['上牌量'].sum().nlargest(8).index
data_top8 = year_brand[year_brand['品牌'].isin(top8)]

plt.figure(figsize=(12, 6))
for brand in top8:
    d = data_top8[data_top8['品牌'] == brand]
    plt.plot(d['年份'], d['份额'], marker='o', label=brand)
plt.title('头部品牌市场份额变化 (2022-2026)', fontsize=14)
plt.xlabel('年份')
plt.ylabel('市场份额 (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/brand_share_trend.png', dpi=150)
plt.show()

# ========== 方向二：城市层级与价格 ==========
print("\n--- 方向二：城市层级与价格 ---")
tier1 = ['北京', '上海', '广州', '深圳']
tier2 = ['成都', '杭州', '武汉', '西安', '苏州', '天津', '南京', '重庆', '郑州', '长沙']

df['城市层级'] = '其他'
df.loc[df['城市'].isin(tier1), '城市层级'] = '一线'
df.loc[df['城市'].isin(tier2), '城市层级'] = '新一线'

tier_price = df.groupby('城市层级')['price_avg'].mean().reindex(['一线', '新一线', '其他'])
print("各城市层级平均购车价格:")
print(tier_price.round(1))

df['价格段'] = pd.cut(df['price_avg'],
                      bins=[0, 10, 15, 20, 30, 50, 100],
                      labels=['10万以下', '10-15万', '15-20万', '20-30万', '30-50万', '50万以上'])

ct = pd.crosstab(df['城市层级'], df['价格段'],
                 values=df['上牌量'], aggfunc='sum', normalize='index') * 100
ct = ct.reindex(['一线', '新一线', '其他'])
ct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
plt.title('不同城市层级的价格段偏好', fontsize=14)
plt.xlabel('城市层级')
plt.ylabel('占比 (%)')
plt.legend(title='价格段', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('../outputs/city_tier_price.png', dpi=150)
plt.show()

# ========== 方向三：春节效应量化 ==========
print("\n--- 方向三：春节效应 ---")
monthly = df.groupby(['年份', '月份'])['上牌量'].sum().reset_index()
monthly['环比'] = monthly['上牌量'].pct_change() * 100

jan_drop = monthly[monthly['月份'] == 1][['年份', '上牌量', '环比']]
print("每年1月销量及环比变化:")
print(jan_drop.round(1))

monthly_season = df.groupby('月份')['上牌量'].mean()
print("\n各月平均销量（季节性）:")
print(monthly_season.round(0))

plt.figure(figsize=(12, 5))
monthly_season.plot(kind='bar', color='coral')
plt.title('新能源汽车销量的月度季节性（多年平均）', fontsize=14)
plt.xlabel('月份')
plt.ylabel('平均上牌量（辆）')
plt.tight_layout()
plt.savefig('../outputs/monthly_seasonality.png', dpi=150)
plt.show()

# ========== 方向四：品牌集中度 ==========
print("\n--- 方向四：品牌集中度 ---")
results = []
for year in sorted(df['年份'].unique()):
    d = df[df['年份'] == year].groupby('品牌')['上牌量'].sum().sort_values(ascending=False)
    total = d.sum()
    cr5 = d.head(5).sum() / total * 100
    cr10 = d.head(10).sum() / total * 100
    n_brands = (d >= 100).sum()
    results.append({'年份': year, 'CR5': cr5, 'CR10': cr10, '活跃品牌数': n_brands})

cr_df = pd.DataFrame(results)
print(cr_df.round(1))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(cr_df['年份'], cr_df['CR5'], alpha=0.6, label='CR5')
ax1.bar(cr_df['年份'], cr_df['CR10'], alpha=0.4, label='CR10')
ax1.set_ylabel('集中度 (%)')
ax1.set_title('新能源市场品牌集中度变化', fontsize=14)

ax2 = ax1.twinx()
ax2.plot(cr_df['年份'], cr_df['活跃品牌数'], 'ro-', label='活跃品牌数')
ax2.set_ylabel('活跃品牌数', color='r')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('../outputs/brand_concentration.png', dpi=150)
plt.show()

print("\n=== 深度 EDA 完成 ===")