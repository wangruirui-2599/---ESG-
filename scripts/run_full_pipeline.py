#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESG Insight Valuator — 完整分析管线
====================================
遵循"数据加载 → 特征工程 → ESG量化 → 异常检测 → 估值 → 因子融合 → 投资建议 → 回测 → 报告"的流程，
支持 --step 参数分步运行，--config 指定配置文件。

使用示例：
  # 运行全部
  python scripts/run_full_pipeline.py --step all

  # 只运行数据加载和特征工程
  python scripts/run_full_pipeline.py --step load --step feature

  # 指定配置目录
  python scripts/run_full_pipeline.py --step all --config config/

  # 只生成报告（需已有中间结果）
  python scripts/run_full_pipeline.py --step report
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.config import load_app_settings, get_config_paths
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.alternative_data import AlternativeDataFusion
from src.data_pipeline.feature_engineering import FeatureEngineeringPipeline
from src.esg_quant.dynamic_weights import DynamicWeightEngine, PolicySignal, create_default_signals
from src.esg_quant.contagion import ContagionAnalyzer
from src.esg_quant.trend_analyzer import ESGTrendAnalyzer
from src.anomaly.predictor import AnomalyPredictor
from src.valuation.scenario import DCFValuator, ScenarioParams
from src.valuation.sentiment_calibration import SentimentCalibrator
from src.fusion.four_factor import FourFactorFusion, RelativeValuationCalculator
from src.advice.probabilistic import ProbabilisticAdvisor
from src.backtest.engine import BacktestEngine
from src.backtest.causal import DIDAnalyzer
from src.reporting.report_generator import ReportGenerator
from src.reporting.visualizer import (
    plot_esg_radar, plot_dcf_scenario_waterfall,
    plot_anomaly_distribution, plot_esg_trend_scatter,
    plot_portfolio_nav, save_all_figures_close,
)

logger = get_logger(__name__)

# 中间结果保存目录
PROCESSED_DIR = Path("data/processed")

# ============================================================================
# 管线步骤
# ============================================================================

STEPS = [
    "load",        # 数据加载
    "feature",     # 特征工程
    "esg",         # ESG量化分析
    "anomaly",     # 异常检测
    "valuation",   # DCF估值
    "fusion",      # 因子融合
    "advice",      # 投资建议
    "backtest",    # 回测验证
    "report",      # 报告生成
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="ESG Insight Valuator — 完整分析管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step", "-s",
        action="append",
        choices=STEPS + ["all"],
        default=[],
        help=f"运行步骤: {', '.join(STEPS + ['all'])} (可多次指定，默认all)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config",
        help="配置文件目录路径 (默认: config/)",
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data",
        help="数据目录路径 (默认: data/)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="输出目录路径 (默认: output/)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        default=True,
        help="跳过已存在的中间结果文件 (默认: True)",
    )
    parser.add_argument(
        "--stock-code",
        type=str,
        default=None,
        help="仅分析指定股票代码",
    )
    return parser.parse_args()


def ensure_dirs(data_dir: str, output_dir: str) -> None:
    """确保必要的目录存在。"""
    for d in [data_dir, output_dir, PROCESSED_DIR,
              f"{output_dir}/figures", f"{output_dir}/reports", "models", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_intermediate(df: pd.DataFrame, name: str) -> None:
    """保存中间结果到 parquet 文件。"""
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"中间结果已保存: {path} ({len(df)} 行)")


def load_intermediate(name: str) -> Optional[pd.DataFrame]:
    """加载中间结果。"""
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"加载中间结果: {path} ({len(df)} 行)")
        return df
    return None


# ============================================================================
# 步骤实现
# ============================================================================

def step_load(args: argparse.Namespace, settings: Any) -> pd.DataFrame:
    """
    Step 1: 数据加载。
    从 CSV 加载财务、ESG、市场数据并合并。
    """
    logger.info("=" * 50)
    logger.info("Step 1: 多源数据加载")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("01_merged_data")
        if cached is not None:
            return cached

    loader = DataLoader(data_dir=args.data_dir)

    # 加载财务数据
    financials = loader.load_csv("raw/financials.csv", schema_name="financials")

    # 加载ESG数据
    esg = loader.load_csv("raw/esg_ratings.csv", schema_name="esg_ratings")

    # 加载市场数据
    market = loader.load_csv("raw/market_data.csv", schema_name="market_data")

    # 合并数据
    merged = financials.merge(esg, on="stock_code", how="left", suffixes=("", "_esg"))
    merged = merged.merge(market, on="stock_code", how="left", suffixes=("", "_mkt"))

    # 日期列同化
    for c in merged.columns:
        if "date" in c.lower() and merged[c].dtype == object:
            merged[c] = pd.to_datetime(merged[c], errors="coerce")

    # 加载另类数据（如存在）
    try:
        sentiment_df = loader.load_csv("external/sentiment_news.csv")
        patent_df = loader.load_csv("external/patents.csv")
        supply_df = loader.load_csv("external/supply_chain.csv")

        fusion = AlternativeDataFusion()
        alt_features = fusion.fuse(sentiment_df, patent_df, supply_df, key_col="stock_code")
        if not alt_features.empty:
            merged = merged.merge(alt_features, on="stock_code", how="left")
            logger.info("另类数据已融合")
    except FileNotFoundError:
        logger.info("另类数据文件不存在，跳过融合")

    save_intermediate(merged, "01_merged_data")
    logger.success(f"Step 1 完成: {len(merged)} 条记录, {len(merged.columns)} 列")
    return merged


def step_feature(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: 特征工程。
    构建滞后特征、滚动统计和行业标准化。
    """
    logger.info("=" * 50)
    logger.info("Step 2: 特征工程")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("02_features")
        if cached is not None:
            return cached

    # 自动识别数值特征列
    exclude = ["stock_code", "industry", "yoy_direction"]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and c not in date_cols
    ]

    # 找到合适的ID列和日期列
    id_col = "stock_code"
    date_col = date_cols[0] if date_cols else "report_date"

    pipeline = FeatureEngineeringPipeline(
        lag_periods=[1, 2, 3, 4],
        rolling_windows=[4, 8, 12],
        standardize_method="zscore_iqr",
        min_industry_size=5,
    )

    df_feat = pipeline.run(
        df, feature_cols=feature_cols[:30],  # 限制特征数，避免维度爆炸
        id_col=id_col, date_col=date_col, industry_col="industry",
    )

    save_intermediate(df_feat, "02_features")
    logger.success(f"Step 2 完成: {len(df_feat.columns)} 列")
    return df_feat


def step_esg(args: argparse.Namespace, df: pd.DataFrame, settings: Any) -> pd.DataFrame:
    """
    Step 3: ESG 量化分析。
    动态权重、行业风险传导、ESG趋势分析。
    """
    logger.info("=" * 50)
    logger.info("Step 3: ESG 量化分析")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("03_esg_quant")
        if cached is not None:
            return cached

    df = df.copy()

    # 3a. 动态权重
    engine = DynamicWeightEngine()
    if settings.esg_weights:
        engine.load_base_weights([w.model_dump() for w in settings.esg_weights])
    # 注册示例信号
    for sig in create_default_signals():
        engine.register_policy_signal(sig)

    df = engine.build_industry_weight_vector(df, industry_col="industry")

    # 3b. 风险传导
    if settings.industry_linkages:
        contagion = ContagionAnalyzer()
        contagion.load_linkages([l.model_dump() for l in settings.industry_linkages])

        # 计算行业风险暴露
        industry_risk = {}
        if "industry" in df.columns and "ESG_total" in df.columns:
            ind_risk = df.groupby("industry")["ESG_total"].mean().to_dict()
            # 风险 = 100 - ESG分
            industry_risk = {k: max(100 - v, 0) for k, v in ind_risk.items()}

        exposure = contagion.compute_exposure(industry_risk)
        logger.info(f"风险传导分析完成: 最受影响的行业={exposure.iloc[0]['industry'] if len(exposure) > 0 else 'N/A'}")

    # 3c. ESG趋势
    trend_analyzer = ESGTrendAnalyzer()
    esg_cols = ["E_score", "S_score", "G_score", "ESG_total"]
    available_esg_cols = [c for c in esg_cols if c in df.columns]
    if available_esg_cols:
        # 选择日期列
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = trend_analyzer.compute_momentum_score(df, score_cols=available_esg_cols, date_col=date_col)
        df = trend_analyzer.classify_trend(df)
        logger.info(f"ESG趋势分类完成: {dict(df['trend_label'].value_counts()) if 'trend_label' in df.columns else {}}")

    save_intermediate(df, "03_esg_quant")
    logger.success("Step 3 完成")
    return df


def step_anomaly(args: argparse.Namespace, df: pd.DataFrame, settings: Any) -> pd.DataFrame:
    """
    Step 4: 财务异常检测。
    训练 LightGBM 模型并预测异常概率。
    """
    logger.info("=" * 50)
    logger.info("Step 4: 财务异常检测")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("04_anomaly")
        if cached is not None:
            return cached

    df = df.copy()

    # 自动构造伪标签（实际部署时应使用真实标签）
    # 这里使用简单规则：净利润<0 或 资产负债率>0.8 标记为异常
    if "is_anomaly" not in df.columns:
        conditions = []
        if "net_profit" in df.columns:
            conditions.append(df["net_profit"] < 0)
        if "debt_ratio" in df.columns:
            conditions.append(df["debt_ratio"] > 0.80)
        if conditions:
            df["is_anomaly"] = pd.concat(conditions, axis=1).any(axis=1).astype(int)
        else:
            df["is_anomaly"] = 0

    n_anomaly = df["is_anomaly"].sum()
    logger.info(f"标签分布: 异常={n_anomaly}, 正常={len(df) - n_anomaly}")

    if n_anomaly < 5:
        logger.warning("异常样本过少，使用随机标签演示")
        df["is_anomaly"] = (np.random.random(len(df)) < 0.15).astype(int)

    # 训练模型
    model_params = {}
    if settings.model_params:
        model_params = settings.model_params.model_dump()

    predictor = AnomalyPredictor(params=model_params)
    X, y = predictor.prepare_data(df, label_col="is_anomaly")

    if y is not None and len(y.unique()) >= 2:
        metrics = predictor.train(X, y, use_cv=True, cv_folds=min(5, n_anomaly))
        logger.info(f"模型训练完成: AUC={metrics.get('auc', 0):.4f}")

        # 预测
        results = predictor.predict(X)
        df["anomaly_probability"] = results["anomaly_probability"].values
        df["risk_level"] = results["risk_level"].values

        # 保存模型
        predictor.save_model("models/anomaly_model.lgb")
    else:
        logger.warning("标签单一，无法训练异常检测模型")
        df["anomaly_probability"] = 0.0
        df["risk_level"] = "低风险"

    save_intermediate(df, "04_anomaly")
    logger.success("Step 4 完成")
    return df


def step_valuation(args: argparse.Namespace, df: pd.DataFrame, settings: Any) -> pd.DataFrame:
    """
    Step 5: DCF 多情景估值。
    对每只股票进行乐观/中性/悲观三情景估值。
    """
    logger.info("=" * 50)
    logger.info("Step 5: DCF 多情景估值")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("05_valuation")
        if cached is not None:
            return cached

    df = df.copy()

    # 加载情景参数
    scenarios = []
    if settings.scenarios:
        for s in settings.scenarios:
            sd = s.model_dump() if hasattr(s, "model_dump") else s
            scenarios.append(ScenarioParams(
                name=sd.get("name", ""),
                revenue_growth=sd.get("revenue_growth", 0.08),
                margin_change=sd.get("margin_change", 0.0),
                wacc=sd.get("wacc", 0.10),
                terminal_growth=sd.get("terminal_growth", 0.025),
                esg_premium=sd.get("esg_premium", 0.0),
                probability=sd.get("probability", 0.33),
            ))

    valuator = DCFValuator(scenarios=scenarios)

    # 为每只股票估值
    results = []
    for _, row in df.iterrows():
        try:
            result = valuator.value_expected(
                stock_code=str(row.get("stock_code", "")),
                base_revenue=row.get("revenue", row.get("total_assets", 100) * 0.5),
                base_margin=row.get("gross_margin", row.get("net_profit", 10) / max(row.get("revenue", 1), 1)),
                current_price=row.get("close_price", row.get("market_cap", 100)),
                total_shares=max(row.get("total_equity", 1), 0.01),
            )
            result["stock_code"] = row.get("stock_code", "")
            results.append(result)
        except Exception as e:
            logger.debug(f"估值失败 [{row.get('stock_code', '?')}]: {e}")

    val_df = pd.DataFrame(results)
    df = df.merge(val_df, on="stock_code", how="left", suffixes=("", "_val"))

    save_intermediate(df, "05_valuation")
    logger.success(f"Step 5 完成: {len(results)} 只股票估值完成")
    return df


def step_fusion(args: argparse.Namespace, df: pd.DataFrame, settings: Any) -> pd.DataFrame:
    """
    Step 6: 四因子融合。
    融合 DCF、相对估值、ESG 和情绪因子。
    """
    logger.info("=" * 50)
    logger.info("Step 6: 四因子融合")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("06_fusion")
        if cached is not None:
            return cached

    df = df.copy()

    # 相对估值计算
    rel_calc = RelativeValuationCalculator()
    if "industry" in df.columns:
        # 假装每股收益 = 净利润/总股本
        if "net_profit" in df.columns and "total_equity" in df.columns:
            df["eps"] = df["net_profit"] / df["total_equity"].replace(0, 1)
            df["bps"] = df["total_equity"] / df["total_equity"].replace(0, 1)
        else:
            df["eps"] = 0.5
            df["bps"] = 5.0

        rel_calc.fit(df, industry_col="industry")
        df["relative_value"] = rel_calc.estimate_dataframe(
            df, industry_col="industry", eps_col="eps", bps_col="bps"
        )
    else:
        df["relative_value"] = df.get("expected_value", df.get("close_price", 0))

    # 情绪校准
    calibrator = SentimentCalibrator()
    if "sentiment_score" in df.columns:
        df = calibrator.calibrate_dataframe(df, value_col="expected_value")
    else:
        df["composite_sentiment"] = 0.0
        df["sentiment_label"] = "中性"
        df["expected_value_calibrated"] = df.get("expected_value", 0)

    # 四因子融合
    fusion_weights = settings.fusion_weights
    fusion = FourFactorFusion(
        dcf_weight=fusion_weights.dcf_weight if fusion_weights else 0.35,
        relative_weight=fusion_weights.relative_weight if fusion_weights else 0.25,
        esg_weight=fusion_weights.esg_weight if fusion_weights else 0.20,
        sentiment_weight=fusion_weights.sentiment_weight if fusion_weights else 0.20,
    )

    df = fusion.fuse_dataframe(
        df,
        dcf_col="expected_value",
        relative_col="relative_value",
        esg_col="expected_value_calibrated",
        sentiment_col="expected_value_calibrated",
        industry_col="industry",
    )

    save_intermediate(df, "06_fusion")
    logger.success("Step 6 完成")
    return df


def step_advice(args: argparse.Namespace, df: pd.DataFrame, settings: Any) -> pd.DataFrame:
    """
    Step 7: 概率投资建议。
    基于期望估值和风险指标生成投资建议。
    """
    logger.info("=" * 50)
    logger.info("Step 7: 概率投资建议")
    logger.info("=" * 50)

    if args.skip_if_exists:
        cached = load_intermediate("07_advice")
        if cached is not None:
            return cached

    df = df.copy()

    advisor = ProbabilisticAdvisor(
        threshold_buy=settings.advice_threshold_buy,
        threshold_sell=settings.advice_threshold_sell,
        confidence_level=settings.confidence_level,
    )

    # 为每只股票生成建议
    results = []
    for _, row in df.iterrows():
        try:
            opt_val = row.get("expected_value", 0) * 1.2
            pes_val = row.get("expected_value", 0) * 0.75
            neu_val = row.get("expected_value", 0)

            result = advisor.evaluate(
                optimistic_value=opt_val,
                neutral_value=neu_val,
                pessimistic_value=pes_val,
                current_price=row.get("current_price", row.get("close_price", neu_val)),
                esg_trend_score=row.get("ESG_total_momentum", 0.0),
                anomaly_probability=row.get("anomaly_probability", 0.0),
                sentiment_score=row.get("composite_sentiment", 0.0),
                esg_trend_label=str(row.get("trend_label", "")),
            )
            result["stock_code"] = row.get("stock_code", "")
            results.append(result)
        except Exception as e:
            logger.debug(f"建议生成失败 [{row.get('stock_code', '?')}]: {e}")

    advice_df = pd.DataFrame(results)
    if "stock_code" in df.columns and "stock_code" in advice_df.columns:
        df = df.merge(advice_df, on="stock_code", how="left", suffixes=("", "_adv"))

    save_intermediate(df, "07_advice")
    # 打印建议分布
    if "advice" in advice_df.columns:
        logger.info(f"建议分布:\n{advice_df['advice'].value_counts().to_string()}")
    logger.success("Step 7 完成")
    return df


def step_backtest(args: argparse.Namespace, df: pd.DataFrame, settings: Any) -> Dict[str, Any]:
    """
    Step 8: 回测验证。
    历史回测 + DID 因果推断。
    """
    logger.info("=" * 50)
    logger.info("Step 8: 回测验证")
    logger.info("=" * 50)

    backtest_results = {}

    # 8a. 历史回测
    try:
        engine = BacktestEngine(
            rebalance_frequency=settings.rebalance_frequency,
        )
        df_prepared = engine.prepare_data(df)
        metrics = engine.run(df_prepared, signal_col="fusion_upside_pct")
        backtest_results = metrics

        if metrics:
            report = engine.generate_report()
            logger.info(f"回测绩效:\n{report}")
    except Exception as e:
        logger.error(f"回测失败: {e}")
        logger.info("可能需要实际的日频数据，跳过回测")

    # 8b. DID 因果推断（如有事件日期数据）
    try:
        if "ESG_total" in df.columns and "ESG_total_lag_4" in df.columns:
            did = DIDAnalyzer()
            did_result = did.run_full_analysis(
                df,
                event_date="2024-03-31",
                outcome_col="close_price" if "close_price" in df.columns else df.select_dtypes(include=[np.number]).columns[0],
            )
            backtest_results["did_analysis"] = did_result.get("conclusion", "")
            logger.info(f"DID结论: {did_result.get('conclusion', '')}")
    except Exception as e:
        logger.error(f"DID分析失败: {e}")

    # 保存回测结果
    import json

    with open(PROCESSED_DIR / "08_backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=2, default=str)

    logger.success("Step 8 完成")
    return backtest_results


def step_report(
    args: argparse.Namespace,
    df: pd.DataFrame,
    backtest_results: Dict[str, Any],
    settings: Any,
) -> None:
    """
    Step 9: 报告生成。
    生成 HTML/Markdown 报告和可视化图表。
    """
    logger.info("=" * 50)
    logger.info("Step 9: 报告生成")
    logger.info("=" * 50)

    generator = ReportGenerator(output_dir=f"{args.output_dir}/reports")

    # 筛选要报告的股票
    if args.stock_code:
        report_stocks = [args.stock_code]
    else:
        # 选择排名前10的股票
        if "fusion_upside_pct" in df.columns:
            top_df = df.nlargest(10, "fusion_upside_pct")
        else:
            top_df = df.head(min(10, len(df)))
        report_stocks = top_df["stock_code"].unique().tolist() if "stock_code" in df.columns else []

    for stock in report_stocks[:10]:  # 最多10份报告
        stock_data = df[df["stock_code"] == stock] if "stock_code" in df.columns else df.head(1)
        if stock_data.empty:
            continue

        row = stock_data.iloc[-1]  # 最新一条记录

        context = generator.build_context(
            stock_code=str(stock),
            industry=str(row.get("industry", "N/A")),
            esg_data={
                "E_score": row.get("E_score", 0),
                "S_score": row.get("S_score", 0),
                "G_score": row.get("G_score", 0),
                "ESG_total": row.get("ESG_total", 0),
                "trend_label": str(row.get("trend_label", "")),
                "ESG_total_momentum": row.get("ESG_total_momentum", 0),
            },
            valuation_data={
                "expected_value": row.get("expected_value", 0),
                "current_price": row.get("current_price", row.get("close_price", 1)),
                "expected_upside_pct": row.get("expected_upside_pct", 0),
                "scenarios": [
                    {"name": "乐观", "probability": 0.25,
                     "intrinsic_value": row.get("expected_value", 0) * 1.2,
                     "upside_pct": (row.get("expected_value", 0) * 1.2 - row.get("close_price", 1))
                                   / max(row.get("close_price", 1), 0.01) * 100},
                    {"name": "中性", "probability": 0.50,
                     "intrinsic_value": row.get("expected_value", 0),
                     "upside_pct": row.get("expected_upside_pct", 0)},
                    {"name": "悲观", "probability": 0.25,
                     "intrinsic_value": row.get("expected_value", 0) * 0.75,
                     "upside_pct": (row.get("expected_value", 0) * 0.75 - row.get("close_price", 1))
                                   / max(row.get("close_price", 1), 0.01) * 100},
                ],
            },
            anomaly_data={
                "anomaly_probability": row.get("anomaly_probability", 0),
                "risk_level": str(row.get("risk_level", "低风险")),
            },
            advice_data={
                "advice": str(row.get("advice", "持有")),
                "confidence": row.get("confidence", 0.5),
                "score": row.get("score", 0),
                "risk_warnings": [f"⚠ 基于模型估算，仅供参考"],
                "key_metrics": {
                    "expected_upside_pct": row.get("expected_upside_pct", 0),
                    "asymmetry_ratio": row.get("asymmetry_ratio", 1.0),
                    "sharpe_approx": row.get("sharpe_approx", 0.0),
                    "esg_trend": str(row.get("trend_label", "")),
                },
            },
            backtest_data=backtest_results if backtest_results else None,
        )

        generator.generate_full_report(context, stock_code=str(stock)[:8])
        logger.info(f"已为 {stock} 生成报告")

    # 批量汇总报告
    if len(df) > 0:
        generator.generate_batch_report(df)

    # 可视化图表
    logger.info("生成可视化图表...")
    out_fig = f"{args.output_dir}/figures"

    try:
        # ESG趋势散点图
        if "ESG_total_momentum" in df.columns and "trend_score" in df.columns:
            plot_esg_trend_scatter(df, output_path=f"{out_fig}/esg_trend_scatter.png")
            save_all_figures_close()

        # 异常分布
        if "anomaly_probability" in df.columns:
            plot_anomaly_distribution(
                df["anomaly_probability"].dropna(),
                output_path=f"{out_fig}/anomaly_distribution.png",
            )
            save_all_figures_close()

        logger.info(f"图表保存至: {out_fig}/")
    except Exception as e:
        logger.error(f"图表生成失败: {e}")

    logger.success("Step 9 完成")


# ============================================================================
# 主入口
# ============================================================================

def main() -> None:
    """主管线入口。"""
    args = parse_args()

    # 初始化日志
    setup_logger(log_level=args.log_level, log_dir="logs")
    logger.info("=" * 60)
    logger.info("ESG Insight Valuator — 启动")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"步骤: {args.step}")
    logger.info("=" * 60)

    # 确保目录
    ensure_dirs(args.data_dir, args.output_dir)

    # 加载配置
    try:
        settings = load_app_settings(args.config)
        logger.info(f"配置加载成功: {settings.project_name} v{settings.version}")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        sys.exit(1)

    # 解析步骤：空列表或无参数 → 全部运行
    if not args.step or "all" in args.step:
        active_steps = list(STEPS)
    else:
        active_steps = [s for s in STEPS if s in args.step]

    logger.info(f"执行步骤: {active_steps}")

    # 逐步执行
    df = pd.DataFrame()
    backtest_results: Dict[str, Any] = {}

    # 自动加载中间结果：如果跳过前几步，尝试加载上一步的缓存数据
    if "load" not in active_steps and "feature" not in active_steps and "esg" not in active_steps:
        # 直接从 anomaly 或更后开始 → 加载 esg 步骤的中间结果
        cached_esg = load_intermediate("03_esg_quant")
        if cached_esg is not None:
            df = cached_esg
            logger.info("已加载 Step 3 (ESG量化) 中间结果，直接进入后续步骤")
        else:
            # 尝试从 CSV 加载
            csv_path = Path("data/processed/03_esg_quant.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.info(f"从CSV加载中间结果: {csv_path} ({len(df)} 行)")
    elif "load" not in active_steps and "feature" not in active_steps:
        # 从 esg 开始
        cached_feat = load_intermediate("02_features")
        if cached_feat is not None:
            df = cached_feat
            logger.info("已加载 Step 2 (特征工程) 中间结果")

    step_funcs = {
        "load": lambda: step_load(args, settings),
        "feature": lambda: step_feature(args, df),
        "esg": lambda: step_esg(args, df, settings),
        "anomaly": lambda: step_anomaly(args, df, settings),
        "valuation": lambda: step_valuation(args, df, settings),
        "fusion": lambda: step_fusion(args, df, settings),
        "advice": lambda: step_advice(args, df, settings),
        "backtest": lambda: step_backtest(args, df, settings),
        "report": lambda: step_report(args, df, backtest_results, settings),
    }

    for step in active_steps:
        if step not in step_funcs:
            logger.warning(f"未知步骤: {step}")
            continue

        try:
            result = step_funcs[step]()

            if step == "load":
                df = result
            elif step == "backtest":
                backtest_results = result
            elif step == "report":
                pass  # report 不返回 DataFrame
            elif isinstance(result, pd.DataFrame) and not result.empty:
                df = result
            else:
                pass  # 保持当前 df

        except KeyboardInterrupt:
            logger.warning("用户中断")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"步骤 [{step}] 执行失败: {e}")
            logger.warning("继续执行后续步骤...")

    logger.info("=" * 60)
    logger.success("🎉 ESG Insight Valuator 管线运行完成！")
    logger.info(f"输出目录: {args.output_dir}/")
    logger.info(f"日志目录: logs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
