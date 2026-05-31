#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实数据导入与适配脚本
======================
将已处理的 ESG 数据（final_complete_data.csv）映射到 EIV 管线的标准 schema，
使得 EIV 管线可以直接从 Step 4 (anomaly) 继续运行。

数据来源: F:/xwechat_files/.../data/data/output/final_complete_data.csv
数据状态: 已完成 Step 1-3 (加载/另类融合/清洗/ESG权重/趋势/传导)

使用方式:
  python scripts/import_real_data.py
  python scripts/run_full_pipeline.py --step anomaly --step valuation --step fusion --step advice --step backtest --step report
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# ============================================================================
# 完整的列名映射表
# ============================================================================

COLUMN_MAPPING = {
    # --- 公司标识 ---
    "company_code": "stock_code",       # 股票代码
    "company_name": "company_name",      # 公司名称（保留）
    "industry": "industry",              # 行业
    "year": "report_year",               # 年份 → 报告年份

    # --- 财务数据 ---
    "revenue": "revenue",                # 营收
    "net_profit": "net_profit",          # 净利润
    "total_assets": "total_assets",      # 总资产
    "total_liabilities": "total_liabilities",  # 总负债
    "total_equity": "total_equity",      # 股东权益

    # --- ESG 评分 ---
    "e_score": "E_score",                # 环境评分
    "s_score": "S_score",                # 社会评分
    "g_score": "G_score",                # 治理评分
    "esg_total_score": "ESG_total",      # ESG综合评分

    # --- 市场 ---
    "stock_price": "close_price",        # 股价 → 收盘价（也是 current_price）

    # --- 另类数据（已融合）---
    "news_count": "news_count",
    "sentiment_score": "sentiment_score",
    "sentiment_std": "sentiment_std",
    "negative_news_ratio": "negative_news_ratio",
    "total_patent_count": "total_patent_count",
    "green_patent_count": "green_patent_count",
    "green_patent_ratio": "green_patent_ratio",
    "supplier_concentration": "supplier_concentration",
    "customer_concentration": "customer_concentration",
    "supply_chain_risk": "supply_chain_risk",

    # --- ESG 动态权重 ---
    "weight_E": "dyn_E_weight",
    "weight_S": "dyn_S_weight",
    "weight_G": "dyn_G_weight",
    "weighted_esg_score": "weighted_esg_score",

    # --- ESG 趋势分析 ---
    "yoy_change": "yoy_change",
    "momentum_score": "ESG_total_momentum",
    "trend_label": "trend_label",

    # --- 风险传导 ---
    "contagion_risk_score": "contagion_risk_score",
    "upstream_risk_exposure": "upstream_risk_exposure",
    "affected_by_industries": "affected_by_industries",
    "affected_by_count": "affected_by_count",
}


def import_and_adapt(
    input_path: str,
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    导入真实数据并映射到 EIV 标准 schema。

    Parameters
    ----------
    input_path : str
        final_complete_data.csv 的路径
    output_dir : str
        中间结果输出目录

    Returns
    -------
    pd.DataFrame
        映射后的 DataFrame
    """
    # 1. 加载数据
    logger.info(f"加载数据: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 2. 列名映射
    df = df.rename(columns=COLUMN_MAPPING)
    mapped_count = sum(1 for v in COLUMN_MAPPING.values() if v in df.columns)
    logger.info(f"列名映射: {mapped_count}/{len(COLUMN_MAPPING)} 列已映射")

    # 3. 派生 EIV 需要的额外列
    # 3a. current_price (同 close_price)
    if "close_price" in df.columns and "current_price" not in df.columns:
        df["current_price"] = df["close_price"]

    # 3b. 财务衍生指标
    if "total_assets" in df.columns and "total_liabilities" in df.columns:
        df["debt_ratio"] = (
            df["total_liabilities"] / df["total_assets"].replace(0, np.nan)
        ).round(4)

    if "net_profit" in df.columns and "total_equity" in df.columns:
        df["roe"] = (
            df["net_profit"] / df["total_equity"].replace(0, np.nan)
        ).round(4)

    if "revenue" in df.columns and "net_profit" in df.columns:
        df["gross_margin"] = (
            df["net_profit"] / df["revenue"].replace(0, np.nan)
        ).round(4)

    # 3c. 构造 report_date（从年份推断）
    if "report_year" in df.columns:
        df["report_date"] = df["report_year"].apply(
            lambda y: f"{int(y)}-12-31" if pd.notna(y) else "2024-12-31"
        )
        df["rating_date"] = df["report_date"]
        df["trade_date"] = df["report_date"]

    # 3d. 相对估值相关
    # 每股收益EPS = 净利润 / 总股本(假设总股本=总资产/股价)
    if "close_price" in df.columns and "total_assets" in df.columns:
        implied_shares = df["total_assets"] / df["close_price"].replace(0, 1)
    else:
        implied_shares = 10.0
    df["eps"] = df["net_profit"] / implied_shares.replace(0, 1)
    df["bps"] = df["total_equity"] / implied_shares.replace(0, 1)
    # PE_TTM 和 PB
    df["pe_ttm"] = (df["close_price"] / df["eps"].replace(0, np.nan)).clip(0, 500)
    df["pb"] = (df["close_price"] / df["bps"].replace(0, np.nan)).clip(0, 50)
    # market_cap (亿元) = 股价 × 总股本
    df["market_cap"] = df["close_price"] * implied_shares

    # 3e. 市场情绪因子（如原数据没有，填充默认值）
    for col in ["composite_sentiment", "sentiment_label",
                "northbound_net_flow", "northbound_avg",
                "margin_balance", "margin_avg",
                "turnover_rate", "avg_turnover"]:
        if col not in df.columns:
            if "rate" in col or "turnover" in col:
                df[col] = 5.0  # 默认换手率5%
            elif "sentiment" in col and "label" not in col:
                df[col] = 0.0
            elif "label" in col:
                df[col] = "中性"
            else:
                df[col] = 0.0

    # 3f. 异常标签（基于多维规则生成，用于演示训练）
    # 评估每家公司是否是"异常"的三个维度:
    #   a. ESG总分在行业后25%
    #   b. ESG动量为负（恶化趋势）
    #   c. 净利润率在全市场后25%
    # 满足至少2个条件即标记为异常
    if "is_anomaly" not in df.columns:
        score_p25 = df["ESG_total"].quantile(0.25)
        margin_p25 = (df["net_profit"] / df["revenue"].replace(0, np.nan)).quantile(0.25)

        cond_a = df["ESG_total"] < score_p25
        cond_b = df.get("ESG_total_momentum", pd.Series(0, index=df.index)) < 0
        cond_c = (df["net_profit"] / df["revenue"].replace(0, np.nan)) < margin_p25

        df["is_anomaly"] = (
            cond_a.astype(int) + cond_b.astype(int) + cond_c.astype(int) >= 2
        ).astype(int)

    # 4. 补充 sentiment_confidence
    if "sentiment_std" in df.columns and "sentiment_confidence" not in df.columns:
        # 标准差越低信心越高
        df["sentiment_confidence"] = np.clip(
            1.0 - df["sentiment_std"].fillna(0.5), 0.1, 1.0
        )

    # 5. 保存中间结果（对应管线 Step 3 完成后的状态）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存为 CSV（管线从 step 4 加载此文件）
    csv_path = Path(output_dir) / "03_esg_quant.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.success(f"数据已保存: {csv_path}")

    # 尝试保存 parquet（可选）
    try:
        parquet_path = Path(output_dir) / "03_esg_quant.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Parquet备份: {parquet_path}")
    except ImportError:
        logger.info("pyarrow未安装，跳过parquet格式")
        logger.info("如需parquet支持: pip install pyarrow")

    # 6. 数据摘要
    logger.info("=" * 50)
    logger.info("📊 数据导入摘要")
    logger.info(f"  公司数: {df['stock_code'].nunique()}")
    logger.info(f"  行业数: {df['industry'].nunique()}")
    logger.info(f"  年份范围: {df['report_year'].min():.0f} - {df['report_year'].max():.0f}")
    logger.info(f"  总行数: {len(df)}")
    logger.info(f"  总列数: {len(df.columns)}")
    logger.info(f"  行业分布: {dict(df['industry'].value_counts())}")

    if "trend_label" in df.columns:
        logger.info(f"  ESG趋势: {dict(df['trend_label'].value_counts())}")

    if "is_anomaly" in df.columns:
        n_anom = df["is_anomaly"].sum()
        logger.info(f"  异常标签: {n_anom} 异常 / {len(df) - n_anom} 正常")

    logger.info("=" * 50)
    logger.info("下一步运行:")
    logger.info("  python scripts/run_full_pipeline.py --step anomaly --step valuation --step fusion --step advice --step backtest --step report")
    logger.info("=" * 50)

    return df


if __name__ == "__main__":
    from src.utils.logger import setup_logger
    setup_logger(log_level="INFO")

    # 数据路径
    DATA_PATH = r"F:\xwechat_files\wxid_ozwd749s5btu22_9dc0\msg\file\2026-05\data\data\output\final_complete_data.csv"

    # 备选：如果已复制到项目目录
    alt_path = "data/raw/final_complete_data.csv"
    if Path(alt_path).exists():
        DATA_PATH = alt_path
        logger.info(f"使用本地副本: {alt_path}")

    import_and_adapt(DATA_PATH, "data/processed")
