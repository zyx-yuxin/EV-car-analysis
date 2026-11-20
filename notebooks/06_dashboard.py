import os
import pandas as pd
import joblib

# ========== 配置路径 ==========
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

# ========== 模型评估指标(通俗版) ==========
metrics_explain = {
    '平均猜错数量 (MAE)': {'value': '22.23 辆', 'desc': '模型预测下月销量时，平均每次偏差约 22 辆车。'},
    '严厉惩罚大错 (RMSE)': {'value': '60.01 辆',
                            'desc': '如果模型出现严重预测失误(如爆款或滞销)，误差会被放大，目前为 60 辆。'},
    '预测靠谱程度 (R²)': {'value': '61.14%',
                          'desc': '模型能解释约 61% 的销量变化规律(满分100%)。在60万大数据量下表现合格。'},
    '是否过拟合': {'value': '轻微', 'desc': '训练集靠谱度76%，测试集61%。模型对老数据很熟，对新数据(2026年)稍有生疏。'}
}

# ========== 特征重要性(中文+业务解释) ==========
feature_explain = [
    {'name': '上牌量(截断)', 'imp': 0.439, 'desc': '🥇 核心因素：当前/历史上的卖车成绩。'},
    {'name': '上月销量', 'imp': 0.248, 'desc': '🥈 核心因素：上个月卖得好，下个月通常也好。'},
    {'name': '品牌排名', 'imp': 0.050, 'desc': '🥉 大品牌自带流量，销量更有保障。'},
    {'name': '城市层级', 'imp': 0.047, 'desc': '一二线城市新能源车接受度更高。'},
    {'name': '省份层级', 'imp': 0.041, 'desc': '所在省份的整体经济水平影响销量。'},
    {'name': '车型累计销量', 'imp': 0.038, 'desc': '该车型历史上的总销量沉淀。'},
    {'name': '品牌累计销量', 'imp': 0.034, 'desc': '该品牌历史上的总销量沉淀。'},
    {'name': '平均价格', 'imp': 0.033, 'desc': '车辆定价区间对销量的影响。'},
    {'name': '是否春节', 'imp': 0.028, 'desc': '春节假期对门店交车的影响。'},
    {'name': '价格段', 'imp': 0.023, 'desc': '10-15万等价格区间的定位影响。'},
    {'name': '是否旺季', 'imp': 0.017, 'desc': '常规旺季对销量的拉动(影响最小)。'}
]

# ========== HTML 模板 ==========
html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>新能源车销量分析总看板</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; color: #333; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        h1 {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 15px; }}
        h2 {{ color: #2980b9; margin-top: 30px; border-left: 4px solid #2980b9; padding-left: 10px; }}
        /* 指标卡片 */
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-top: 20px; }}
        .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; border-left: 5px solid #3498db; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 16px; color: #7f8c8d; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .metric-card .desc {{ font-size: 13px; color: #666; line-height: 1.5; }}
        /* 进度条(R²) */
        .progress-bar {{ width: 100%; background-color: #ecf0f1; border-radius: 4px; overflow: hidden; height: 12px; margin-top: 5px; }}
        .progress-fill {{ height: 100%; background-color: #2ecc71; text-align: right; padding-right: 5px; color: white; font-size: 10px; line-height: 12px; }}
        /* 特征表格 */
        .feature-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        .feature-table th, .feature-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .feature-table th {{ background-color: #f8f9fa; color: #2c3e50; }}
        .feature-table tr:nth-child(even) {{ background-color: #fcfcfc; }}
        /* 图表区 */
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .chart-card {{ border: 1px solid #eee; border-radius: 8px; padding: 15px; text-align: center; }}
        .chart-card img {{ max-width: 100%; border-radius: 4px; }}
        .links a {{ display: block; margin: 10px 0; color: #2980b9; text-decoration: none; font-size: 15px; }}
        .links a:hover {{ text-decoration: underline; }}
        .footer {{ text-align: center; margin-top: 40px; color: #95a5a6; font-size: 13px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🚗 新能源车销量预测与特征分析看板</h1>

    <h2>📈 模型表现</h2>
    <div class="metrics-grid">
        {"".join(f'''
        <div class="metric-card">
            <h3>{k}</h3>
            <div class="value">{v['value']}</div>
            <div class="desc">{v['desc']}</div>
        </div>''' for k, v in metrics_explain.items())}

        <!-- R² 进度条特化处理 -->
        <div class="metric-card" style="border-left-color: #2ecc71;">
            <h3>预测靠谱程度可视化</h3>
            <div class="value">61.14%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 61.14%;"></div>
            </div>
            <div class="desc" style="margin-top: 8px;">绿色越满代表预测越准（满分100%）。</div>
        </div>
    </div>

    <h2>🏆 核心发现：什么因素决定了销量？</h2>
    <table class="feature-table">
        <tr><th>排名</th><th>特征名称</th><th>重要性</th><th>业务含义解读</th></tr>
        {"".join(f'<tr><td>{i + 1}</td><td><b>{f["name"]}</b></td><td>{f["imp"]}</td><td>{f["desc"]}</td></tr>' for i, f in enumerate(feature_explain))}
    </table>

    <h2>🖼️ 预测效果与静态图表</h2>
    <div class="charts-grid">
        <div class="chart-card">
            <h3>特征重要性图</h3>
            <img src="feature_importance.png" alt="特征重要性">
        </div>
        <div class="chart-card">
            <h3>预测值 vs 真实值</h3>
            <img src="pred_vs_true.png" alt="预测值 vs 真实值">
        </div>
    </div>

    <h2>🔗 交互式业务图表（点击探索）</h2>
    <div class="links">
        <a href="province_top15_bar.html" target="_blank">📍 查看各省份销量 TOP 15</a>
        <a href="brand_top10_timeline.html" target="_blank">🏆 查看品牌 TOP 10 动态排名（可拖时间轴）</a>
        <a href="monthly_trend_interactive.html" target="_blank">📈 查看月度销量走势（可缩放）</a>
        <a href="price_segment_pie.html" target="_blank">🥧 查看各价格段占比</a>
    </div>

    <div class="footer">
        <p>新能源车数据分析项目 · 数据周期：2022-01 ~ 2026-12 · 共 60万+ 条记录</p>
    </div>
</div>
</body>
</html>
"""

# ========== 保存文件 ==========
output_path = os.path.join(output_dir, 'index.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅看板已生成: {output_path}")