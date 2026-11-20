
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载数据 ==========
df = pd.read_csv('../data/cleaned_nev_data.csv')
print(f"数据形状: {df.shape}")
print(f"\n时间范围: {df['年份'].min()}-{df['月份'].min()} 到 {df['年份'].max()}-{df['月份'].max()}")
print(f"\n品牌数: {df['品牌'].nunique()}")
print(f"车型数: {df['车型'].nunique()}")
print(f"城市数: {df['城市'].nunique()}")

# ========== 2. 全局统计 ==========
print("\n=== 上牌量统计 ===")
print(df['上牌量'].describe())
print(f"\n价格区间: {df['price_avg'].min():.1f} - {df['price_avg'].max():.1f} 万")

# ========== 3. 月度总销量走势 ==========
monthly = df.groupby(['年份', '月份'])['上牌量'].sum().reset_index()
monthly['年月'] = monthly['年份'].astype(str) + '-' + monthly['月份'].astype(str).str.zfill(2)

plt.figure(figsize=(14, 5))
plt.plot(monthly['年月'], monthly['上牌量'], marker='o', linewidth=2)
plt.xticks(rotation=45)
plt.title('月度新能源车总上牌量走势', fontsize=14)
plt.xlabel('年月')
plt.ylabel('上牌量（辆）')
plt.tight_layout()
plt.savefig('../outputs/monthly_trend.png', dpi=150)
plt.show()

# ========== 4. 品牌销量 TOP 20 ==========
brand_sales = df.groupby('品牌')['上牌量'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
brand_sales.plot(kind='bar')
plt.title('品牌总上牌量 TOP 20', fontsize=14)
plt.xlabel('品牌')
plt.ylabel('上牌量（辆）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/brand_top20.png', dpi=150)
plt.show()

# ========== 5. 城市销量 TOP 20 ==========
city_sales = df.groupby('城市')['上牌量'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
city_sales.plot(kind='bar')
plt.title('城市总上牌量 TOP 20', fontsize=14)
plt.xlabel('城市')
plt.ylabel('上牌量（辆）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/city_top20.png', dpi=150)
plt.show()

# ========== 6. 价格区间 vs 销量 ==========
df['价格段'] = pd.cut(df['price_avg'],
                      bins=[0, 10, 15, 20, 30, 50, 100],
                      labels=['10万以下', '10-15万', '15-20万', '20-30万', '30-50万', '50万以上'])

price_seg = df.groupby('价格段')['上牌量'].sum()

plt.figure(figsize=(10, 5))
price_seg.plot(kind='bar', color='orange')
plt.title('各价格段总上牌量', fontsize=14)
plt.xlabel('价格段')
plt.ylabel('上牌量（辆）')
plt.tight_layout()
plt.savefig('../outputs/price_segment.png', dpi=150)
plt.show()

# ========== 7. 省份热力图数据（给后面 pyecharts 用）==========
province_sales = df.groupby('省份')['上牌量'].sum().sort_values(ascending=False)
print("\n=== 省份销量 TOP 10 ===")
print(province_sales.head(10))

print("\n=== EDA 完成 ===")