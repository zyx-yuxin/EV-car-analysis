
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 正常显示负号


# ========== 1. 加载数据 ==========
df = pd.read_csv('../data/features_with_target.csv', encoding='utf-8-sig')
print(f"原始数据: {df.shape}")

# ========== 2. 数据预处理 ==========
# 去掉目标变量为NaN的行（最后一个月没有下个月数据）
df = df.dropna(subset=['next_month_sales'])
print(f"去掉NaN后: {df.shape}")

# 编码分类变量
le_price = LabelEncoder()
le_province = LabelEncoder()
df['price_segment_enc'] = le_price.fit_transform(df['price_segment'].astype(str))
df['province_tier_enc'] = le_province.fit_transform(df['province_tier'].astype(str))

# 选择特征
feature_cols = [
    'price_avg', '上牌量_capped', 'is_spring_festival', 'is_peak_season',
    'city_tier', 'price_segment_enc', 'province_tier_enc',
    'brand_cumulative_sales', 'brand_rank', 'model_cumulative_sales',
    'last_month_sales'
]

X = df[feature_cols]
y = df['next_month_sales']

print(f"\n特征列表: {feature_cols}")
print(f"特征形状: {X.shape}")

# ========== 3. 划分训练集/测试集 ==========
# 按时间划分：用2022-2025训练，2026测试（模拟真实预测场景）
train_mask = df['年份'] <= 2025
test_mask = df['年份'] == 2026

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ========== 4. 训练 XGBoost ==========
print("\n正在训练 XGBoost 模型...")
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ========== 5. 预测与评估 ==========
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n=== 模型评估结果 ===")
print(f"{'指标':<15} {'训练集':<15} {'测试集':<15}")
print(f"{'RMSE':<15} {train_rmse:<15.2f} {test_rmse:<15.2f}")
print(f"{'MAE':<15} {train_mae:<15.2f} {test_mae:<15.2f}")
print(f"{'R²':<15} {train_r2:<15.4f} {test_r2:<15.4f}")

# ========== 6. 特征重要性 ==========
importance = model.feature_importances_
importance_df = pd.DataFrame({
    '特征': feature_cols,
    '重要性': importance
}).sort_values('重要性', ascending=False)

# ========== 新增：英文特征名映射为中文 ==========
name_map = {
    '上牌量_capped': '上牌量(截断)',
    'last_month_sales': '上月销量',
    'brand_rank': '品牌排名',
    'city_tier': '城市层级',
    'province_tier_enc': '省份层级',
    'model_cumulative_sales': '车型累计销量',
    'brand_cumulative_sales': '品牌累计销量',
    'price_avg': '平均价格',
    'is_spring_festival': '是否春节',
    'price_segment_enc': '价格段',
    'is_peak_season': '是否旺季'
}
importance_df['特征'] = importance_df['特征'].map(name_map)
# ==================================================

print("\n=== 特征重要性排名 ===")
print(importance_df.to_string(index=False))


# 画特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(importance_df['特征'][::-1], importance_df['重要性'][::-1])
plt.title('XGBoost 特征重要性', fontsize=14)
plt.xlabel('重要性')
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png', dpi=150)
plt.show()

# ========== 7. 预测值 vs 真实值散点图 ==========
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_test, alpha=0.3, s=5)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', linewidth=2)
plt.xlabel('真实值（下月销量）')
plt.ylabel('预测值（下月销量）')
plt.title(f'预测值 vs 真实值 (测试集 R²={test_r2:.4f})', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/pred_vs_true.png', dpi=150)
plt.show()

# ========== 8. 保存模型 ==========
import joblib
joblib.dump(model, '../outputs/xgb_model.pkl')
print("\n✅ 模型已保存: outputs/xgb_model.pkl")

print("\n=== 建模完成 ===")