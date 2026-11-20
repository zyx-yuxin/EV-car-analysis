import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import os

# ========== 配置 ==========
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

# ========== 加载数据 ==========
df = pd.read_csv('../data/cleaned_nev_data.csv')
print(f"数据加载完成: {df.shape}")

# ========== 通用保存函数（完全离线） ==========
def save_plotly(fig, filename):
    """用 plotly.offline.plot 保存，JS 完全内嵌，双击即可打开"""
    html_path = os.path.join(output_dir, filename)
    plot(fig, filename=html_path, auto_open=False, include_plotlyjs=True)
    # include_plotlyjs=True 会把整个 plotly.js 库写进 HTML（约3MB），完全离线
    print(f"✅ {filename} 已生成")

# ========== 图1：省份 TOP 15 条形图 ==========
print("正在生成省份 TOP 15 条形图...")
province_sales = df.groupby('省份')['上牌量'].sum().sort_values().tail(15).reset_index()
province_sales.columns = ['省份', '上牌量']

fig1 = px.bar(
    province_sales,
    x='上牌量',
    y='省份',
    orientation='h',
    title='省份新能源车累计上牌量 TOP 15',
    color='上牌量',
    color_continuous_scale='Blues',
    labels={'上牌量': '上牌量（辆）', '省份': '省份'}
)
fig1.update_layout(height=600, width=1000, yaxis={'categoryorder': 'total ascending'})
save_plotly(fig1, 'province_top15_bar.html')

# ========== 图2：品牌 TOP 10 动态排名（时间轴动画） ==========
print("正在生成品牌动态排名图...")
year_brand = df.groupby(['年份', '品牌'])['上牌量'].sum().reset_index()
# 每年取TOP10
top10_by_year = []
for year in sorted(year_brand['年份'].unique()):
    d = year_brand[year_brand['年份'] == year].sort_values('上牌量', ascending=False).head(10)
    d['排名'] = range(1, len(d) + 1)
    top10_by_year.append(d)
top10_df = pd.concat(top10_by_year)

fig2 = px.bar(
    top10_df,
    x='上牌量',
    y='品牌',
    color='上牌量',
    animation_frame='年份',
    animation_group='品牌',
    range_x=[0, top10_df['上牌量'].max() * 1.1],
    title='品牌新能源车销量 TOP 10 动态排名',
    labels={'上牌量': '上牌量（辆）', '品牌': '品牌', '年份': '年份'}
)
fig2.update_layout(height=600, width=1200)
# 播放按钮样式
fig2.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1500
fig2.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
save_plotly(fig2, 'brand_top10_timeline.html')

# ========== 图3：月度走势（可缩放） ==========
print("正在生成月度走势图...")
monthly = df.groupby(['年份', '月份'])['上牌量'].sum().reset_index()
monthly['年月'] = monthly['年份'].astype(str) + '-' + monthly['月份'].astype(str).str.zfill(2)
monthly = monthly.sort_values('年月')

fig3 = px.line(
    monthly,
    x='年月',
    y='上牌量',
    title='月度新能源车总上牌量走势',
    labels={'年月': '月份', '上牌量': '上牌量（辆）'},
    markers=True
)
fig3.update_layout(height=500, width=1200)
fig3.update_xaxes(rangeslider_visible=True)  # 底部滑动缩放条
save_plotly(fig3, 'monthly_trend_interactive.html')

# ========== 图4：价格段占比环形图 ==========
print("正在生成价格段饼图...")
df['价格段'] = pd.cut(
    df['price_avg'],
    bins=[0, 10, 15, 20, 30, 50, 100],
    labels=['10万以下', '10-15万', '15-20万', '20-30万', '30-50万', '50万以上']
)
price_dist = df.groupby('价格段')['上牌量'].sum().reset_index()

fig4 = px.pie(
    price_dist,
    names='价格段',
    values='上牌量',
    title='各价格段销量占比',
    hole=0.4,  # 环形图（0.4是中间空心比例）
)
fig4.update_traces(textinfo='label+percent', textposition='outside')
fig4.update_layout(height=600, width=800)
save_plotly(fig4, 'price_segment_pie.html')

print("\n=== 全部完成 ===")
print("所有图表已生成，双击 HTML 文件即可查看（完全离线，无需网络）")