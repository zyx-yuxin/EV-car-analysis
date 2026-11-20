# 🚗 新能源车销量分析项目

## 项目简介
基于 2022-2026 年新能源车上牌数据（60万+条），通过 XGBoost 构建销量预测模型，并搭建交互式可视化看板。

## 技术栈
- Python 3.13
- Pandas / Scikit-learn / XGBoost
- Plotly（交互式可视化）
- Matplotlib（静态图表）

## 项目结构
- `notebooks/03_viz_plotly.py`：4张交互式图表（Plotly）
- `notebooks/04_feature_engineering.py`：特征工程
- `notebooks/05_model.py`：XGBoost 销量预测模型
- `notebooks/06_dashboard.py`：总看板生成

## 模型效果
- 测试集 R²：0.6114
- 核心特征：历史销量惯性（上牌量、上月销量）

## 运行方式
1. 安装依赖：`pip install -r requirements.txt`
2. 运行特征工程：`python notebooks/04_feature_engineering.py`
3. 运行建模：`python notebooks/05_model.py`
4. 运行可视化：`python notebooks/03_viz_plotly.py`
5. 生成看板：`python notebooks/06_dashboard.py`
6. 打开 `outputs/index.html` 查看总看板


##可直接打开 `outputs/index.html` 查看总看板



##本项目仅供学习交流使用。
