#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例数据生成脚本
================
生成用于测试和演示的模拟数据集，覆盖 EIV 系统所需的所有数据类别。

生成文件：
  - data/raw/financials.csv        财务基本面数据
  - data/raw/esg_ratings.csv        ESG评分数据
  - data/raw/market_data.csv        市场行情数据
  - data/raw/industry.csv           行业分类
  - data/external/sentiment_news.csv 舆情新闻（另类数据）
  - data/external/patents.csv        专利数据（另类数据）
  - data/external/supply_chain.csv   供应链数据（另类数据）
  - data/external/northbound_flow.csv 北向资金
  - data/external/margin_trading.csv  两融数据

使用方式：
  python scripts/generate_sample_data.py
  python scripts/generate_sample_data.py --n-stocks 500 --n-quarters 20
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42
np.random.seed(SEED)


# ============================================================================
# 行业列表与股票
# ============================================================================

INDUSTRIES = [
    "石油石化", "煤炭", "有色金属", "钢铁", "基础化工",
    "建筑材料", "机械设备", "电力设备", "汽车", "家用电器",
    "食品饮料", "纺织服装", "轻工制造", "商贸零售", "社会服务",
    "医药生物", "银行", "非银金融", "房地产",
    "电子", "计算机", "传媒", "通信",
    "公用事业", "交通运输", "环保",
    "国防军工", "建筑装饰", "农林牧渔",
]


def generate_stock_codes(n: int) -> list:
    """生成模拟股票代码列表。"""
    codes = []
    for i in range(n):
        exchange = "60" if i % 3 == 0 else "00"
        num = str(i + 1).zfill(4)
        codes.append(f"{exchange}{num}.{'SH' if exchange == '60' else 'SZ'}")
    return codes


def assign_industries(stock_codes: list) -> pd.DataFrame:
    """为每只股票随机分配行业。"""
    n = len(stock_codes)
    industries = np.random.choice(INDUSTRIES, size=n)
    return pd.DataFrame({
        "stock_code": stock_codes,
        "industry": industries,
    })


# ============================================================================
# 1. 财务数据
# ============================================================================

def generate_financials(
    stock_codes: list,
    industry_map: pd.DataFrame,
    n_quarters: int = 16,
) -> pd.DataFrame:
    """
    生成模拟财务基本面数据。

    为每只股票生成多个季度的财务指标，具有合理的行业差异和时间趋势。
    """
    records = []
    base_date = datetime(2021, 3, 31)

    for code in stock_codes:
        industry = industry_map[industry_map["stock_code"] == code]["industry"].values[0]

        # 行业特征参数
        industry_params = {
            "银行": {"revenue_base": 500, "margin": 0.35, "growth": 0.06},
            "石油石化": {"revenue_base": 300, "margin": 0.08, "growth": 0.04},
            "钢铁": {"revenue_base": 150, "margin": 0.05, "growth": 0.03},
            "房地产": {"revenue_base": 200, "margin": 0.12, "growth": 0.02},
            "食品饮料": {"revenue_base": 100, "margin": 0.20, "growth": 0.10},
            "医药生物": {"revenue_base": 80, "margin": 0.18, "growth": 0.12},
            "计算机": {"revenue_base": 50, "margin": 0.15, "growth": 0.15},
            "公用事业": {"revenue_base": 120, "margin": 0.10, "growth": 0.05},
        }
        params = industry_params.get(industry, {"revenue_base": 100, "margin": 0.12, "growth": 0.08})

        revenue_base = params["revenue_base"] * np.random.uniform(0.5, 2.0)
        margin_base = params["margin"] + np.random.normal(0, 0.03)
        growth = params["growth"] + np.random.normal(0, 0.02)

        for q in range(n_quarters):
            date = base_date + timedelta(days=q * 90)
            # 营收带季节性和趋势
            seasonal = 1 + 0.05 * np.sin(q * np.pi / 2)
            revenue = revenue_base * (1 + growth) ** (q / 4) * seasonal
            revenue *= np.random.uniform(0.9, 1.1)

            margin = np.clip(margin_base + np.random.normal(0, 0.02), -0.10, 0.50)
            net_profit = revenue * margin
            total_assets = revenue * np.random.uniform(1.5, 3.0)
            total_equity = total_assets * np.random.uniform(0.3, 0.6)

            records.append({
                "stock_code": code,
                "报告期": date.strftime("%Y-%m-%d"),
                "营业总收入": round(revenue, 2),
                "净利润": round(net_profit, 2),
                "资产总计": round(total_assets, 2),
                "股东权益": round(total_equity, 2),
                "经营活动现金流": round(net_profit * np.random.uniform(0.5, 1.5), 2),
                "ROE": round(net_profit / max(total_equity, 1), 4),
                "资产负债率": round((total_assets - total_equity) / max(total_assets, 1), 4),
                "营收增速": round(growth + np.random.normal(0, 0.03), 4),
                "毛利率": round(np.clip(margin + np.random.normal(0, 0.05), 0.05, 0.7), 4),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["stock_code", "报告期"]).reset_index(drop=True)
    print(f"  → financials.csv: {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    return df


# ============================================================================
# 2. ESG 评分数据
# ============================================================================

def generate_esg_ratings(
    stock_codes: list,
    industry_map: pd.DataFrame,
    n_quarters: int = 16,
) -> pd.DataFrame:
    """
    生成模拟 ESG 评分数据。

    各行业有不同的 ESG 基线，并随时间呈现趋势性变化。
    """
    records = []
    base_date = datetime(2021, 3, 31)

    # 行业ESG基线
    esg_baselines = {
        "石油石化": (55, 60, 65), "煤炭": (50, 58, 62),
        "钢铁": (52, 60, 63), "基础化工": (55, 62, 65),
        "公用事业": (60, 65, 68), "环保": (75, 70, 72),
        "食品饮料": (65, 68, 70), "医药生物": (68, 72, 75),
        "银行": (60, 65, 78), "非银金融": (58, 63, 75),
        "计算机": (62, 68, 72), "电子": (60, 65, 70),
        "房地产": (55, 60, 68), "建筑装饰": (58, 62, 65),
        "交通运输": (60, 63, 67), "商贸零售": (62, 66, 68),
    }

    for code in stock_codes:
        industry = industry_map[industry_map["stock_code"] == code]["industry"].values[0]
        e_base, s_base, g_base = esg_baselines.get(industry, (60, 65, 68))

        # 每只股票带独立噪音和趋势
        e_trend = np.random.uniform(-0.2, 1.5)  # 每季度环境分变化
        s_trend = np.random.uniform(-0.1, 1.0)
        g_trend = np.random.uniform(-0.1, 0.8)

        for q in range(n_quarters):
            date = base_date + timedelta(days=q * 90)
            e = np.clip(e_base + e_trend * q + np.random.normal(0, 3), 0, 100)
            s = np.clip(s_base + s_trend * q + np.random.normal(0, 3), 0, 100)
            g = np.clip(g_base + g_trend * q + np.random.normal(0, 3), 0, 100)
            total = e * 0.35 + s * 0.30 + g * 0.35

            records.append({
                "stock_code": code,
                "rating_date": date.strftime("%Y-%m-%d"),
                "E_score": round(e, 2),
                "S_score": round(s, 2),
                "G_score": round(g, 2),
                "ESG_total": round(total, 2),
                "rating_agency": np.random.choice(["MSCI", "中财绿金", "商道融绿", "Wind ESG"]),
                "industry": industry,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["stock_code", "rating_date"]).reset_index(drop=True)
    print(f"  → esg_ratings.csv: {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    return df


# ============================================================================
# 3. 市场行情数据
# ============================================================================

def generate_market_data(
    stock_codes: list,
    n_days: int = 500,
) -> pd.DataFrame:
    """
    生成模拟日频市场行情数据。
    """
    records = []
    base_date = datetime(2023, 1, 1)

    for code in stock_codes:
        price = np.random.uniform(5, 100)
        base_mcap = price * np.random.uniform(5, 500)

        for d in range(n_days):
            date = base_date + timedelta(days=d)
            if date.weekday() >= 5:  # 跳过周末
                continue

            daily_return = np.random.normal(0.0005, 0.02)
            price *= (1 + daily_return)
            price = max(price, 1.0)

            pe = np.random.uniform(8, 60)
            pb = np.random.uniform(0.5, 8)
            turnover = np.random.uniform(0.1, 15)
            volume = np.random.uniform(100, 10000)

            records.append({
                "stock_code": code,
                "交易日期": date.strftime("%Y-%m-%d"),
                "收盘价": round(price, 2),
                "总市值": round(price * base_mcap / (price if d == 0 else price), 2),
                "市盈率": round(pe, 2),
                "市净率": round(pb, 2),
                "换手率": round(turnover, 2),
                "成交量": round(volume, 2),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["stock_code", "交易日期"]).reset_index(drop=True)
    print(f"  → market_data.csv: {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    return df


# ============================================================================
# 4. 另类数据 — 舆情
# ============================================================================

def generate_sentiment_news(
    stock_codes: list,
    n_articles: int = 1000,
) -> pd.DataFrame:
    """生成模拟舆情新闻数据。"""
    templates_pos = [
        "公司{code}发布{year}年度报告，营收同比增长{rate}%，超出市场预期。",
        "以绿色创新引领发展——{code}成功研发低碳{product}技术，获得行业标杆认证。",
        "{code}获ESG评级上调至AA级，外资持续增持。",
        "公司稳健运营，{code}连续{year2}年入选可持续发展百强企业。",
    ]
    templates_neg = [
        "环保督查通报：{code}旗下子公司因违规排放被罚款{amt}万元。",
        "员工维权事件发酵，{code}面临劳动纠纷诉讼。",
        "财报显示{code}净利润大幅下滑{rate}%，引发市场担忧。",
        "监管问询函：{code}关联交易信息披露不充分。",
    ]

    records = []
    for _ in range(n_articles):
        code = np.random.choice(stock_codes)
        is_positive = np.random.random() > 0.35
        templates = templates_pos if is_positive else templates_neg
        text = np.random.choice(templates).format(
            code=code[:6],
            year=np.random.randint(2020, 2025),
            year2=np.random.randint(3, 10),
            rate=np.random.randint(5, 40),
            product=np.random.choice(["光伏", "储能", "氢能", "碳捕获", "节能"]),
            amt=np.random.randint(50, 500),
        )

        records.append({
            "stock_code": code,
            "date": (datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 700))).strftime("%Y-%m-%d"),
            "content": text,
            "source": np.random.choice(["财经网", "证券时报", "新浪财经", "Wind", "东方财富"]),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    print(f"  → sentiment_news.csv: {len(df)} 行")
    return df


# ============================================================================
# 5. 另类数据 — 专利
# ============================================================================

GREEN_IPC_LIST = [
    "B01D53/00", "C02F1/00", "B09B3/00", "F03D1/00",
    "H01L31/00", "H01M8/00", "H01M10/00", "C10L5/00",
    "B60L50/00", "Y02E10/00",
]
NORMAL_IPC_LIST = [
    "G06F17/00", "A61K31/00", "H04L9/00", "G01N33/00",
    "B23K26/00", "C07D401/00", "E21B43/00", "H05K3/00",
]


def generate_patents(stock_codes: list, n_patents: int = 500) -> pd.DataFrame:
    """生成模拟专利数据。"""
    records = []
    for _ in range(n_patents):
        code = np.random.choice(stock_codes)
        is_green = np.random.random() < 0.25
        ipc = np.random.choice(GREEN_IPC_LIST if is_green else NORMAL_IPC_LIST)

        records.append({
            "stock_code": code,
            "patent_id": f"CN{np.random.randint(10000000, 99999999)}B",
            "ipc_code": ipc,
            "application_date": (
                datetime(2018, 1, 1) + timedelta(days=np.random.randint(0, 2000))
            ).strftime("%Y-%m-%d"),
            "title": f"一种{'绿色' if is_green else '改进的'}"
                     f"{np.random.choice(['装置', '方法', '系统', '材料'])}",
        })

    df = pd.DataFrame(records)
    print(f"  → patents.csv: {len(df)} 行, 绿色占比≈{df['ipc_code'].isin(GREEN_IPC_LIST).mean():.0%}")
    return df


# ============================================================================
# 6. 另类数据 — 供应链
# ============================================================================

def generate_supply_chain(stock_codes: list, n_edges: int = 300) -> pd.DataFrame:
    """生成模拟供应链关系。"""
    records = []
    for _ in range(n_edges):
        supplier = np.random.choice(stock_codes)
        customer = np.random.choice(stock_codes)
        if supplier == customer:
            continue

        records.append({
            "supplier_code": supplier,
            "customer_code": customer,
            "transaction_amount": round(np.random.uniform(0.1, 50), 2),
            "revenue_share": round(np.random.uniform(0.01, 0.30), 4),
            "relationship": np.random.choice(["核心供应商", "一般供应商", "战略合作伙伴"]),
        })

    df = pd.DataFrame(records)
    print(f"  → supply_chain.csv: {len(df)} 条关系")
    return df


# ============================================================================
# 7. 北向资金 & 两融
# ============================================================================

def generate_northbound_flow(stock_codes: list, n_days: int = 500) -> pd.DataFrame:
    """生成模拟北向资金流向数据。"""
    records = []
    base_date = datetime(2023, 1, 1)
    for code in stock_codes:
        flow_base = np.random.uniform(-2, 5)
        for d in range(n_days):
            date = base_date + timedelta(days=d)
            if date.weekday() >= 5:
                continue
            daily_flow = flow_base + np.random.normal(0, 3)
            records.append({
                "stock_code": code,
                "trade_date": date.strftime("%Y-%m-%d"),
                "net_flow_100m": round(daily_flow, 2),
                "cumulative_holding_pct": round(np.random.uniform(0.5, 10), 2),
            })
    df = pd.DataFrame(records)
    print(f"  → northbound_flow.csv: {len(df)} 行")
    return df


def generate_margin_trading(stock_codes: list, n_days: int = 500) -> pd.DataFrame:
    """生成模拟两融余额数据。"""
    records = []
    base_date = datetime(2023, 1, 1)
    for code in stock_codes:
        margin_base = np.random.uniform(1, 50)
        for d in range(n_days):
            date = base_date + timedelta(days=d)
            if date.weekday() >= 5:
                continue
            margin_balance = margin_base + np.random.normal(0, margin_base * 0.03)
            short_balance = margin_balance * np.random.uniform(0.01, 0.15)
            records.append({
                "stock_code": code,
                "trade_date": date.strftime("%Y-%m-%d"),
                "margin_balance_100m": round(max(margin_balance, 0.1), 2),
                "short_balance_100m": round(max(short_balance, 0.01), 2),
            })
    df = pd.DataFrame(records)
    print(f"  → margin_trading.csv: {len(df)} 行")
    return df


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 EIV 示例数据")
    parser.add_argument("--n-stocks", type=int, default=100, help="股票数量 (默认100)")
    parser.add_argument("--n-quarters", type=int, default=16, help="季度数 (默认16=4年)")
    parser.add_argument("--n-days", type=int, default=500, help="行情天数 (默认500)")
    parser.add_argument("--data-dir", type=str, default="data", help="数据根目录")
    args = parser.parse_args()

    print("=" * 50)
    print("ESG Insight Valuator — 示例数据生成")
    print(f"  股票数: {args.n_stocks}")
    print(f"  季度数: {args.n_quarters}")
    print(f"  行情天数: {args.n_days}")
    print("=" * 50)

    # 创建目录
    for sub in ["raw", "external"]:
        Path(args.data_dir, sub).mkdir(parents=True, exist_ok=True)

    # 生成股票列表和行业
    stock_codes = generate_stock_codes(args.n_stocks)
    industry_map = assign_industries(stock_codes)

    # 保存行业分类
    industry_map.to_csv(Path(args.data_dir, "raw", "industry.csv"), index=False, encoding="utf-8")
    print("✅ 行业分类")

    # 1. 财务数据
    print("\n📊 生成财务数据...")
    df = generate_financials(stock_codes, industry_map, args.n_quarters)
    df.to_csv(Path(args.data_dir, "raw", "financials.csv"), index=False, encoding="utf-8")

    # 2. ESG数据
    print("\n🌱 生成ESG评分...")
    df = generate_esg_ratings(stock_codes, industry_map, args.n_quarters)
    df.to_csv(Path(args.data_dir, "raw", "esg_ratings.csv"), index=False, encoding="utf-8")

    # 3. 市场行情
    print("\n📈 生成市场行情...")
    df = generate_market_data(stock_codes, args.n_days)
    df.to_csv(Path(args.data_dir, "raw", "market_data.csv"), index=False, encoding="utf-8")

    # 4. 舆情
    print("\n📰 生成舆情新闻...")
    df = generate_sentiment_news(stock_codes)
    df.to_csv(Path(args.data_dir, "external", "sentiment_news.csv"), index=False, encoding="utf-8")

    # 5. 专利
    print("\n🔬 生成专利数据...")
    df = generate_patents(stock_codes)
    df.to_csv(Path(args.data_dir, "external", "patents.csv"), index=False, encoding="utf-8")

    # 6. 供应链
    print("\n🔗 生成供应链数据...")
    df = generate_supply_chain(stock_codes)
    df.to_csv(Path(args.data_dir, "external", "supply_chain.csv"), index=False, encoding="utf-8")

    # 7. 北向资金
    print("\n💹 生成北向资金和两融数据...")
    df = generate_northbound_flow(stock_codes, args.n_days)
    df.to_csv(Path(args.data_dir, "external", "northbound_flow.csv"), index=False, encoding="utf-8")

    df = generate_margin_trading(stock_codes, args.n_days)
    df.to_csv(Path(args.data_dir, "external", "margin_trading.csv"), index=False, encoding="utf-8")

    print("\n" + "=" * 50)
    print("✅ 所有示例数据已生成！")
    print(f"   数据目录: {args.data_dir}/")
    print(f"\n   下一步: python scripts/run_full_pipeline.py --step all")
    print("=" * 50)


if __name__ == "__main__":
    main()
