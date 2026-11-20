##数据清洗
#1.指导价由区间改为最低，最高和均价
#2.查找缺失值
#3.查找异常值

#1.确定指导价格是否都是区间价格
# import pandas as pd
# df = pd.read_excel(r'D:\PYstudy\EV-car-analysis\data\新能源汽车上牌量202206-202609.xlsx')
# print(df.columns.tolist())
# print(df.shape)
# print(df['指导价(万元)'].head(20).tolist())
# print(df['指导价(万元)'].isnull().sum())

#2.正式清洗

import pandas as pd
import numpy as np
import re
import os


def clean_price(df, price_col='指导价(万元)'):
    """
    把 '31.69-41.79' 拆成最低/最高/均价
    支持：区间 'xx-yy'、单个数字 'xx'、空值
    """
    def parse(price_str):
        if pd.isna(price_str):
            return np.nan, np.nan, np.nan
        s = str(price_str).strip()
        # 区间
        m = re.match(r'^(\d+\.?\d*)\s*[-~]\s*(\d+\.?\d*)$', s)
        if m:
            low, high = float(m.group(1)), float(m.group(2))
            return low, high, (low + high) / 2
        # 单个数字
        m = re.match(r'^(\d+\.?\d*)$', s)
        if m:
            v = float(m.group(1))
            return v, v, v
        return np.nan, np.nan, np.nan

    print("正在拆分指导价...")
    parsed = df[price_col].apply(parse)
    df['price_low'] = [p[0] for p in parsed]
    df['price_high'] = [p[1] for p in parsed]
    df['price_avg'] = [p[2] for p in parsed]

    ok = df['price_avg'].notna().sum()
    print(f"  价格解析成功: {ok}/{len(df)} ({ok/len(df)*100:.1f}%)")

    return df


def handle_missing(df):
    """处理缺失值：看看哪些列有缺失，再决定怎么处理"""
    print("\n=== 缺失值统计 ===")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        print("  无缺失值，数据完整")
    else:
        print(miss)
        # 上牌量/时间字段缺失 → 直接删除
        essential = ['年份', '月份', '上牌量']
        before = len(df)
        df = df.dropna(subset=essential)
        print(f"  关键字段缺失删除: {before} → {len(df)} 行")
        # 价格缺失 → 用同品牌均价填充
        if df['price_avg'].isnull().sum() > 0:
            df['price_avg'] = df.groupby('品牌')['price_avg'].transform(
                lambda x: x.fillna(x.median())
            )
            df['price_low'] = df.groupby('品牌')['price_low'].transform(
                lambda x: x.fillna(x.median())
            )
            df['price_high'] = df.groupby('品牌')['price_high'].transform(
                lambda x: x.fillna(x.median())
            )
            print(f"  价格缺失用品牌中位数填充完成")
    return df


def handle_outliers(df, col='上牌量', method='cap'):
    """
    处理异常值
    method='cap'   → 用分位数截断（保留样本，适合时序/分组数据）
    method='drop'  → 直接删除（样本多时用）
    """
    print(f"\n正在处理 {col} 异常值...")
    print(f"  处理前: min={df[col].min()}, max={df[col].max()}, "
          f"median={df[col].median():.0f}, p99={df[col].quantile(0.99):.0f}")

    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)

    if method == 'cap':
        df[col + '_capped'] = df[col].clip(lower=lower, upper=upper)
        changed = (df[col] != df[col + '_capped']).sum()
        print(f"  用 1%~99% 分位数截断: 影响 {changed} 行")
        return df
    else:
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"  删除异常值: {before} → {len(df)} 行")
        return df


def main():
    input_path = r'D:\PYstudy\EV-car-analysis\data\新能源汽车上牌量202206-202609.xlsx'
    output_path = r'D:\PYstudy\EV-car-analysis\data\cleaned_nev_data.csv'

    print("=== 读取数据 ===")
    df = pd.read_excel(input_path)
    print(f"原始: {df.shape[0]} 行 × {df.shape[1]} 列")

    # 1. 拆分价格
    df = clean_price(df)

    # 2. 缺失值
    df = handle_missing(df)

    # 3. 异常值（截断，保留样本）
    df = handle_outliers(df, col='上牌量', method='cap')

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n=== 完成 ===")
    print(f"输出: {output_path} ({df.shape[0]} 行 × {df.shape[1]} 列)")


if __name__ == '__main__':
    main()


