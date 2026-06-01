"""
市场情绪校准模块
================
通过四大情绪因子对 DCF 估值结果进行市场情绪校准，
使估值更贴近当前市场定价环境。

四因子：
  1. 北向资金流向 — 外资对A股的配置情绪
  2. 融资融券余额 — 杠杆资金的看多/看空情绪
  3. 换手率 — 市场交投活跃度（过度交易 = 过度乐观/恐慌）
  4. 舆情情感 — 新闻/社交媒体文本情感得分

校准公式：
  adjusted_value = base_value × (1 + Σ(因子值 × 因子权重 × 因子系数))
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 情绪因子定义
# ============================================================================

class SentimentFactor:
    """单个情绪因子"""

    def __init__(
        self,
        name: str,
        weight: float,
        coefficient: float = 1.0,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """
        初始化情绪因子。

        Parameters
        ----------
        name : str
            因子名称
        weight : float
            因子在总分中的权重 (0~1)
        coefficient : float
            缩放系数
        clip_range : tuple
            因子值截断范围
        """
        self.name = name
        self.weight = weight
        self.coefficient = coefficient
        self.clip_range = clip_range


# ============================================================================
# 情绪校准器
# ============================================================================

class SentimentCalibrator:
    """
    市场情绪校准器。

    整合北向资金、两融、换手率、舆情四大因子，
    计算情绪得分并对 DCF 估值结果进行校准。

    Attributes
    ----------
    factors : list of SentimentFactor
        情绪因子列表
    market_sentiment_cache : dict
        市场情绪缓存
    """

    def __init__(
        self,
        northbound_weight: float = 0.30,
        margin_weight: float = 0.25,
        turnover_weight: float = 0.20,
        sentiment_text_weight: float = 0.25,
    ) -> None:
        """
        初始化情绪校准器。

        Parameters
        ----------
        northbound_weight : float
            北向资金权重
        margin_weight : float
            两融权重
        turnover_weight : float
            换手率权重
        sentiment_text_weight : float
            舆情文本权重
        """
        total = northbound_weight + margin_weight + turnover_weight + sentiment_text_weight
        self.factors = [
            SentimentFactor("northbound", northbound_weight / total, clip_range=(-0.5, 0.5)),
            SentimentFactor("margin", margin_weight / total, clip_range=(-0.5, 0.5)),
            SentimentFactor("turnover", turnover_weight / total, clip_range=(-0.3, 0.3)),
            SentimentFactor("sentiment_text", sentiment_text_weight / total, clip_range=(-0.5, 0.5)),
        ]
        logger.info(
            f"SentimentCalibrator 初始化: "
            f"北向={northbound_weight:.0%}, 两融={margin_weight:.0%}, "
            f"换手率={turnover_weight:.0%}, 舆情={sentiment_text_weight:.0%}"
        )

    def compute_northbound_factor(
        self, net_flow: float, avg_daily_flow: float
    ) -> float:
        """
        计算北向资金情绪因子。

        算法：净流入相对日均成交的比例 → Z-score → Sigmoid归一化

        Parameters
        ----------
        net_flow : float
            北向资金当日净流入（亿元）
        avg_daily_flow : float
            过去20日日均净流入（亿元）

        Returns
        -------
        float
            情绪因子值 [-0.5, 0.5]
        """
        if avg_daily_flow == 0:
            return 0.0
        ratio = (net_flow - avg_daily_flow) / (abs(avg_daily_flow) + 1e-8)
        # Sigmoid 归一化
        score = 2.0 / (1.0 + np.exp(-ratio)) - 1.0
        return np.clip(score * 0.5, -0.5, 0.5)

    def compute_margin_factor(
        self, margin_balance: float, avg_balance: float
    ) -> float:
        """
        计算两融情绪因子。

        融资余额增长 → 看多情绪；余额快速下降 → 恐慌情绪

        Parameters
        ----------
        margin_balance : float
            当前融资余额（亿元）
        avg_balance : float
            过去20日均值（亿元）

        Returns
        -------
        float
            情绪因子值 [-0.5, 0.5]
        """
        if avg_balance == 0:
            return 0.0
        change_pct = (margin_balance - avg_balance) / avg_balance
        # 温和上涨最优（+5%），极端值惩罚
        optimal = 0.05
        score = np.exp(-((change_pct - optimal) ** 2) / 0.02) * 2 - 1
        return np.clip(score * 0.5, -0.5, 0.5)

    def compute_turnover_factor(self, turnover_rate: float, avg_turnover: float) -> float:
        """
        计算换手率情绪因子。

        异常高换手 → 情绪过热（惩罚）；适度换手 → 健康（中性）

        Parameters
        ----------
        turnover_rate : float
            当前换手率 (%)
        avg_turnover : float
            历史均值 (%)

        Returns
        -------
        float
            情绪因子值 [-0.3, 0.3]
        """
        if avg_turnover == 0:
            return 0.0
        ratio = turnover_rate / avg_turnover
        if ratio <= 1.0:
            score = 0.0  # 正常偏低 → 中性
        elif ratio <= 2.0:
            score = -(ratio - 1.0) * 0.15  # 轻度惩罚
        else:
            score = -0.15 - (ratio - 2.0) * 0.05  # 重度惩罚
        return np.clip(score, -0.3, 0.3)

    def compute_sentiment_text_factor(
        self, sentiment_score: float, confidence: float = 0.5
    ) -> float:
        """
        基于舆情情感得分计算情绪因子。

        Parameters
        ----------
        sentiment_score : float
            情感得分 [-1, 1]，来自 SentimentAnalyzer
        confidence : float
            情感置信度 [0, 1]

        Returns
        -------
        float
            情绪因子值 [-0.5, 0.5]
        """
        # 置信度加权的情感得分
        weighted = sentiment_score * confidence
        return np.clip(weighted * 0.5, -0.5, 0.5)

    def compute_composite_sentiment(
        self,
        northbound_net_flow: float = 0.0,
        northbound_avg: float = 0.0,
        margin_balance: float = 0.0,
        margin_avg: float = 0.0,
        turnover_rate: float = 0.0,
        avg_turnover: float = 0.0,
        sentiment_score: float = 0.0,
        sentiment_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        计算综合市场情绪得分。

        Parameters
        ----------
        northbound_net_flow : float
            北向资金净流入
        northbound_avg : float
            北向20日均值
        margin_balance : float
            两融余额
        margin_avg : float
            两融20日均值
        turnover_rate : float
            当前换手率
        avg_turnover : float
            历史均值换手率
        sentiment_score : float
            舆情情感得分
        sentiment_confidence : float
            舆情置信度

        Returns
        -------
        dict
            包含各因子得分和综合得分的字典
        """
        # 计算各因子
        northbound_score = self.compute_northbound_factor(
            northbound_net_flow, northbound_avg
        )
        margin_score = self.compute_margin_factor(margin_balance, margin_avg)
        turnover_score = self.compute_turnover_factor(turnover_rate, avg_turnover)
        text_score = self.compute_sentiment_text_factor(
            sentiment_score, sentiment_confidence
        )

        # 加权汇总
        scores = {
            "northbound": northbound_score,
            "margin": margin_score,
            "turnover": turnover_score,
            "sentiment_text": text_score,
        }

        composite = sum(
            scores[f.name] * f.weight * f.coefficient
            for f in self.factors
        )

        result = {
            **{f"{k}_score": round(v, 4) for k, v in scores.items()},
            "composite_sentiment": round(composite, 4),
            "sentiment_label": self._get_sentiment_label(composite),
        }
        return result

    @staticmethod
    def _get_sentiment_label(score: float) -> str:
        """根据综合得分返回情绪标签。"""
        if score > 0.15:
            return "积极乐观"
        elif score > 0.05:
            return "偏乐观"
        elif score > -0.05:
            return "中性"
        elif score > -0.15:
            return "偏悲观"
        else:
            return "悲观恐慌"

    def calibrate_valuation(
        self, base_value: float, composite_sentiment: float
    ) -> float:
        """
        对基础估值进行情绪校准。

        公式: adjusted = base_value × (1 + composite_sentiment)

        Parameters
        ----------
        base_value : float
            基础DCF估值
        composite_sentiment : float
            综合情绪得分

        Returns
        -------
        float
            情绪校准后的估值
        """
        adjusted = base_value * (1 + composite_sentiment)
        logger.debug(
            f"估值校准: {base_value:.2f} × (1 + {composite_sentiment:.4f}) = {adjusted:.2f}"
        )
        return adjusted

    def calibrate_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str = "expected_value",
    ) -> pd.DataFrame:
        """
        对 DataFrame 中的估值列进行情绪校准。

        需要 DataFrame 包含情绪相关数据列：
        northbound_net_flow, northbound_avg, margin_balance, margin_avg,
        turnover_rate, avg_turnover, sentiment_score, sentiment_confidence

        Parameters
        ----------
        df : pd.DataFrame
            待校准数据
        value_col : str
            估值列名

        Returns
        -------
        pd.DataFrame
            添加了情绪校准列的数据表
        """
        df = df.copy()

        sentiments = []
        calibrated_values = []

        for _, row in df.iterrows():
            result = self.compute_composite_sentiment(
                northbound_net_flow=row.get("northbound_net_flow", 0.0),
                northbound_avg=row.get("northbound_avg", 0.0),
                margin_balance=row.get("margin_balance", 0.0),
                margin_avg=row.get("margin_avg", 0.0),
                turnover_rate=row.get("turnover_rate", 0.0),
                avg_turnover=row.get("avg_turnover", 0.0),
                sentiment_score=row.get("sentiment_score", 0.0),
                sentiment_confidence=row.get("sentiment_confidence", 0.5),
            )
            sentiments.append(result)

            base_val = row.get(value_col, 0.0)
            calibrated_values.append(
                self.calibrate_valuation(base_val, result["composite_sentiment"])
            )

        df["composite_sentiment"] = [s["composite_sentiment"] for s in sentiments]
        df["sentiment_label"] = [s["sentiment_label"] for s in sentiments]
        df[f"{value_col}_calibrated"] = [
            round(v, 2) for v in calibrated_values
        ]

        logger.info(
            f"批量情绪校准完成: {len(df)} 条记录, "
            f"平均情绪={df['composite_sentiment'].mean():.4f}"
        )
        return df


# ============================================================================
# 便捷函数
# ============================================================================

def quick_sentiment_calibration(
    northbound_flow: float = 0.0,
    margin_balance: float = 0.0,
    turnover: float = 0.0,
    sentiment_score: float = 0.0,
) -> float:
    """
    便捷函数：快速计算综合情绪得分。

    Parameters
    ----------
    northbound_flow : float
        北向净流入
    margin_balance : float
        两融余额
    turnover : float
        换手率
    sentiment_score : float
        舆情得分

    Returns
    -------
    float
        综合情绪得分
    """
    calibrator = SentimentCalibrator()
    result = calibrator.compute_composite_sentiment(
        northbound_net_flow=northbound_flow,
        northbound_avg=0.0,
        margin_balance=margin_balance,
        margin_avg=0.0,
        turnover_rate=turnover,
        avg_turnover=0.0,
        sentiment_score=sentiment_score,
        sentiment_confidence=0.5,
    )
    return result["composite_sentiment"]
