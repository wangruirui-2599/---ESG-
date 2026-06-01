"""
概率投资建议引擎
================
基于多情景概率加权的估值结果，生成量化的投资建议。

核心逻辑：
  1. 三情景（乐观/中性/悲观）动态概率赋值
  2. 计算期望估值与当前价格的偏离度
  3. 量化上下行风险不对称性
  4. 输出买入/持有/卖出建议及置信度

建议规则：
  - 期望上行 > 15%（买入阈值）且下行风险可控 → 买入
  - 期望上行 < -10%（卖出阈值）→ 卖出
  - 其他 → 持有
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 建议输出类型
# ============================================================================

ADVICE_TYPES = {
    "strong_buy": "强烈买入",
    "buy": "买入",
    "hold": "持有",
    "sell": "卖出",
    "strong_sell": "强烈卖出",
}


# ============================================================================
# 概率建议引擎
# ============================================================================

class ProbabilisticAdvisor:
    """
    概率投资建议引擎。

    基于期望估值、上下行风险和置信区间，
    输出概率化的投资建议。

    Attributes
    ----------
    threshold_buy : float
        买入阈值（期望上行比例）
    threshold_sell : float
        卖出阈值（期望下行比例）
    confidence_level : float
        置信水平
    """

    def __init__(
        self,
        threshold_buy: float = 0.15,
        threshold_sell: float = -0.10,
        confidence_level: float = 0.95,
    ) -> None:
        """
        初始化建议引擎。

        Parameters
        ----------
        threshold_buy : float
            买入阈值，期望上行>此值建议买入
        threshold_sell : float
            卖出阈值，期望上行<此值建议卖出
        confidence_level : float
            VaR计算的置信水平
        """
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.confidence_level = confidence_level
        logger.info(
            f"ProbabilisticAdvisor 初始化: buy>{threshold_buy:.0%}, "
            f"sell<{threshold_sell:.0%}, CL={confidence_level:.0%}"
        )

    def assign_scenario_probabilities(
        self,
        esg_trend_score: float = 0.0,
        anomaly_probability: float = 0.0,
        sentiment_score: float = 0.0,
        base_probabilities: Optional[List[float]] = None,
    ) -> List[float]:
        """
        基于多维度信号动态调整三情景概率。

        调整逻辑：
        - ESG趋势向好 → 乐观概率上调，悲观下调
        - 异常概率高 → 悲观概率上调，乐观下调
        - 情绪积极 → 乐观概率上调
        - 最终归一化确保和为1

        Parameters
        ----------
        esg_trend_score : float
            ESG趋势分数（正值=改善，负值=恶化）
        anomaly_probability : float
            财务异常概率 (0~1)
        sentiment_score : float
            综合市场情绪 (-1~1)

        base_probabilities : list of float, optional
            基础先验概率 [乐观, 中性, 悲观]

        Returns
        -------
        list of float
            调整后的三情景概率
        """
        if base_probabilities is None:
            base_probabilities = [0.25, 0.50, 0.25]

        opt, neu, pes = base_probabilities

        # ESG趋势调整：正向趋势 → 乐观+，趋势分数归一化
        esg_adj = np.clip(esg_trend_score / 20.0, -0.15, 0.15)
        opt += esg_adj
        pes -= esg_adj

        # 异常风险调整：高异常概率 → 悲观+
        anomaly_adj = np.clip(anomaly_probability * 0.30, -0.15, 0.15)
        pes += anomaly_adj
        opt -= anomaly_adj
        neu -= anomaly_adj * 0.5

        # 市场情绪调整：积极情绪 → 乐观+
        sentiment_adj = np.clip(sentiment_score * 0.20, -0.15, 0.15)
        opt += sentiment_adj
        pes -= sentiment_adj

        # 确保非负
        opt = max(opt, 0.05)
        neu = max(neu, 0.15)
        pes = max(pes, 0.05)

        # 归一化
        total = opt + neu + pes
        probabilities = [opt / total, neu / total, pes / total]

        logger.debug(
            f"情景概率调整: ESG趋势={esg_trend_score:.2f}, "
            f"异常={anomaly_probability:.2f}, 情绪={sentiment_score:.2f} → "
            f"[{probabilities[0]:.2%}, {probabilities[1]:.2%}, {probabilities[2]:.2%}]"
        )
        return probabilities

    def compute_expected_metrics(
        self,
        optimistic_value: float,
        neutral_value: float,
        pessimistic_value: float,
        current_price: float,
        probabilities: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        计算期望估值和风险指标。

        Parameters
        ----------
        optimistic_value : float
            乐观情景估值
        neutral_value : float
            中性情景估值
        pessimistic_value : float
            悲观情景估值
        current_price : float
            当前股价
        probabilities : list of float, optional
            三情景概率，默认使用先验值

        Returns
        -------
        dict
            期望估值指标
        """
        if probabilities is None:
            probabilities = [0.25, 0.50, 0.25]

        values = [optimistic_value, neutral_value, pessimistic_value]

        # 期望值
        expected_value = sum(v * p for v, p in zip(values, probabilities))

        # 标准差
        variance = sum(p * (v - expected_value) ** 2 for v, p in zip(values, probabilities))
        std_dev = np.sqrt(variance)

        # VaR (Value at Risk)
        z_score = 1.645  # 95% 置信
        if self.confidence_level == 0.99:
            z_score = 2.326
        var_95 = expected_value - z_score * std_dev

        # 期望上行空间
        if current_price > 0:
            expected_upside = (expected_value - current_price) / current_price
        else:
            expected_upside = 0.0

        # 上下行风险不对称
        upside_potential = max(optimistic_value - current_price, 0)
        downside_risk = max(current_price - pessimistic_value, 0)
        if downside_risk > 0:
            asymmetry_ratio = upside_potential / downside_risk
        else:
            asymmetry_ratio = float("inf") if upside_potential > 0 else 1.0

        # 夏普比率近似（期望超额收益/波动）
        risk_free_rate = 0.03
        sharpe_approx = (
            (expected_upside - risk_free_rate) / max(std_dev / current_price, 0.01)
            if current_price > 0 else 0.0
        )

        return {
            "expected_value": round(expected_value, 2),
            "std_dev": round(std_dev, 2),
            "var_95": round(var_95, 2),
            "expected_upside_pct": round(expected_upside * 100, 2),
            "upside_potential": round(upside_potential, 2),
            "downside_risk": round(downside_risk, 2),
            "asymmetry_ratio": round(asymmetry_ratio, 2),
            "sharpe_approx": round(sharpe_approx, 4),
            "scenario_probabilities": [round(p, 3) for p in probabilities],
        }

    def generate_advice(
        self,
        expected_metrics: Dict[str, Any],
        anomaly_probability: float = 0.0,
        esg_trend_label: str = "",
    ) -> Dict[str, Any]:
        """
        基于期望指标生成投资建议。

        Parameters
        ----------
        expected_metrics : dict
            compute_expected_metrics() 的输出
        anomaly_probability : float
            财务异常概率
        esg_trend_label : str
            ESG趋势标签

        Returns
        -------
        dict
            投资建议
        """
        upside = expected_metrics["expected_upside_pct"] / 100.0
        asymmetry = expected_metrics["asymmetry_ratio"]
        sharpe = expected_metrics["sharpe_approx"]
        var_95 = expected_metrics["var_95"]

        # 多维度评分
        score = 0.0

        # 1. 期望上行得分
        if upside > 0.30:
            score += 3.0
        elif upside > self.threshold_buy:
            score += 2.0
        elif upside > 0.05:
            score += 0.5
        elif upside < self.threshold_sell:
            score -= 3.0
        elif upside < -0.05:
            score -= 1.5

        # 2. 不对称性得分（上行 > 下行 越多越好）
        if asymmetry > 3.0:
            score += 1.5
        elif asymmetry > 2.0:
            score += 1.0
        elif asymmetry < 0.5:
            score -= 1.5

        # 3. 夏普比率得分
        if sharpe > 2.0:
            score += 1.0
        elif sharpe < 0:
            score -= 1.0

        # 4. ESG趋势惩罚/奖励
        if "恶化" in esg_trend_label:
            score -= 0.5
        elif "改善" in esg_trend_label:
            score += 0.5

        # 5. 异常风险惩罚
        if anomaly_probability > 0.5:
            score -= 2.0
        elif anomaly_probability > 0.3:
            score -= 1.0

        # 确定建议类型
        if score >= 3.5:
            advice_type = "strong_buy"
        elif score >= 1.5:
            advice_type = "buy"
        elif score >= -1.5:
            advice_type = "hold"
        elif score >= -3.5:
            advice_type = "sell"
        else:
            advice_type = "strong_sell"

        # 置信度
        confidence = min(abs(score) / 5.0, 1.0)

        result = {
            "advice": ADVICE_TYPES[advice_type],
            "advice_code": advice_type,
            "score": round(score, 2),
            "confidence": round(confidence, 4),
            "key_metrics": {
                "expected_upside_pct": expected_metrics["expected_upside_pct"],
                "asymmetry_ratio": expected_metrics["asymmetry_ratio"],
                "sharpe_approx": expected_metrics["sharpe_approx"],
                "var_95": expected_metrics["var_95"],
                "anomaly_probability": round(anomaly_probability, 4),
                "esg_trend": esg_trend_label,
            },
            "risk_warnings": self._generate_risk_warnings(
                anomaly_probability, esg_trend_label, upside, asymmetry
            ),
        }

        return result

    def _generate_risk_warnings(
        self,
        anomaly_probability: float,
        esg_trend_label: str,
        upside: float,
        asymmetry: float,
    ) -> List[str]:
        """
        生成风险提示。

        Parameters
        ----------
        anomaly_probability : float
            异常概率
        esg_trend_label : str
            ESG趋势
        upside : float
            上行空间
        asymmetry : float
            不对称比

        Returns
        -------
        list of str
            风险提示列表
        """
        warnings = []

        if anomaly_probability > 0.5:
            warnings.append(f"⚠ 财务异常概率高达 {anomaly_probability:.0%}，建议深入尽调")
        elif anomaly_probability > 0.3:
            warnings.append(f"⚡ 财务异常概率 {anomaly_probability:.0%}，存在一定风险")

        if "恶化" in esg_trend_label:
            warnings.append(f"📉 ESG趋势: {esg_trend_label}，可能面临监管和声誉风险")
        elif "波动" in esg_trend_label:
            warnings.append(f"📊 ESG趋势: {esg_trend_label}，不确定性较高")

        if upside < -0.15:
            warnings.append(f"🔻 期望下行空间 {upside:.0%}，估值偏高")

        if asymmetry < 0.8:
            warnings.append(f"⚖ 风险收益不对称，下行风险大于上行空间")

        if not warnings:
            warnings.append("✅ 未检测到显著风险信号")

        return warnings

    def evaluate(
        self,
        optimistic_value: float,
        neutral_value: float,
        pessimistic_value: float,
        current_price: float,
        esg_trend_score: float = 0.0,
        anomaly_probability: float = 0.0,
        sentiment_score: float = 0.0,
        esg_trend_label: str = "",
        base_probabilities: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        一站式评估：情景概率 → 期望指标 → 投资建议。

        Parameters
        ----------
        optimistic_value : float
            乐观估值
        neutral_value : float
            中性估值
        pessimistic_value : float
            悲观估值
        current_price : float
            当前股价
        esg_trend_score : float
            ESG趋势分数
        anomaly_probability : float
            异常概率
        sentiment_score : float
            市场情绪得分
        esg_trend_label : str
            ESG趋势标签
        base_probabilities : list of float, optional
            先验概率

        Returns
        -------
        dict
            完整评估结果
        """
        # Step 1: 动态情景概率
        probabilities = self.assign_scenario_probabilities(
            esg_trend_score, anomaly_probability, sentiment_score, base_probabilities
        )

        # Step 2: 期望指标
        metrics = self.compute_expected_metrics(
            optimistic_value, neutral_value, pessimistic_value,
            current_price, probabilities,
        )

        # Step 3: 投资建议
        advice = self.generate_advice(metrics, anomaly_probability, esg_trend_label)

        return {**metrics, **advice}

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        opt_col: str = "optimistic_value",
        neu_col: str = "neutral_value",
        pes_col: str = "pessimistic_value",
        price_col: str = "current_price",
        esg_score_col: str = "ESG_total",
        anomaly_col: str = "anomaly_probability",
        sentiment_col: str = "composite_sentiment",
        esg_trend_col: str = "trend_label",
    ) -> pd.DataFrame:
        """
        对 DataFrame 进行批量投资评估。

        Parameters
        ----------
        df : pd.DataFrame
            包含各情景估值和相关指标的数据表
        opt_col : str
            乐观估值列
        neu_col : str
            中性估值列
        pes_col : str
            悲观估值列
        price_col : str
            当前股价列
        esg_score_col : str
            ESG评分别
        anomaly_col : str
            异常概率列
        sentiment_col : str
            情绪得分列
        esg_trend_col : str
            ESG趋势标签列

        Returns
        -------
        pd.DataFrame
            投资建议汇总表
        """
        results = []
        for _, row in df.iterrows():
            try:
                result = self.evaluate(
                    optimistic_value=row.get(opt_col, row.get("expected_value", 0)),
                    neutral_value=row.get(neu_col, row.get("expected_value", 0)),
                    pessimistic_value=row.get(pes_col, row.get("expected_value", 0)),
                    current_price=row.get(price_col, 0),
                    esg_trend_score=row.get("ESG_total_momentum", 0.0),
                    anomaly_probability=row.get(anomaly_col, 0.0),
                    sentiment_score=row.get(sentiment_col, 0.0),
                    esg_trend_label=str(row.get(esg_trend_col, "")),
                )
                result["stock_code"] = row.get("stock_code", "")
                results.append(result)
            except Exception as e:
                logger.error(f"评估失败 [{row.get('stock_code', '?')}]: {e}")

        result_df = pd.DataFrame(results)
        logger.info(
            f"批量评估完成: {len(result_df)} 只股票\n"
            f"  建议分布: {dict(result_df['advice'].value_counts())}"
        )
        return result_df


# ============================================================================
# 便捷函数
# ============================================================================

def quick_advice(
    dcf_value: float,
    current_price: float,
    esg_score: float = 50.0,
    anomaly_prob: float = 0.0,
) -> Dict[str, Any]:
    """
    便捷函数：快速生成投资建议。

    Parameters
    ----------
    dcf_value : float
        DCF估值（中性情景）
    current_price : float
        当前股价
    esg_score : float
        ESG评分
    anomaly_prob : float
        异常概率

    Returns
    -------
    dict
        投资建议
    """
    advisor = ProbabilisticAdvisor()
    # 简化：中性±20%作为乐观/悲观
    return advisor.evaluate(
        optimistic_value=dcf_value * 1.20,
        neutral_value=dcf_value,
        pessimistic_value=dcf_value * 0.75,
        current_price=current_price,
        esg_trend_score=(esg_score - 50) / 5,
        anomaly_probability=anomaly_prob,
    )
