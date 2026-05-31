"""
四因子融合模型
==============
将四大估值因子（DCF绝对估值、相对估值、ESG评分、市场情绪）
通过行业动态权重加权融合，输出最终综合估值。

融合公式：
  final_value = w1 × dcf_value + w2 × relative_value + w3 × esg_adjusted + w4 × sentiment_adjusted

行业动态权重：
  不同行业对各因子的依赖程度不同：
  - 消费行业：DCF权重大（现金流稳定），相对估值辅助
  - 科技行业：情绪权重较高（成长预期驱动）
  - 金融行业：相对估值权重高（PB/PE可比性强）
  - 重资产行业：ESG权重高（环境风险大）
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 融合引擎
# ============================================================================

class FourFactorFusion:
    """
    四因子融合引擎。

    将 DCF、相对估值、ESG、情绪四个维度的估值进行加权融合，
    权重可通过行业模板动态调整。

    Attributes
    ----------
    default_weights : dict
        默认四因子权重 {"dcf": w1, "relative": w2, "esg": w3, "sentiment": w4}
    industry_weights : dict
        行业权重覆盖 {行业名: {"dcf": w1, ...}}
    """

    def __init__(
        self,
        dcf_weight: float = 0.35,
        relative_weight: float = 0.25,
        esg_weight: float = 0.20,
        sentiment_weight: float = 0.20,
    ) -> None:
        """
        初始化四因子融合引擎。

        Parameters
        ----------
        dcf_weight : float
            DCF估值权重
        relative_weight : float
            相对估值权重
        esg_weight : float
            ESG因子权重
        sentiment_weight : float
            市场情绪权重
        """
        total = dcf_weight + relative_weight + esg_weight + sentiment_weight
        self.default_weights = {
            "dcf": dcf_weight / total,
            "relative": relative_weight / total,
            "esg": esg_weight / total,
            "sentiment": sentiment_weight / total,
        }
        self.industry_weights: Dict[str, Dict[str, float]] = self._init_industry_weights()
        logger.info(
            f"FourFactorFusion 初始化: "
            f"DCF={self.default_weights['dcf']:.0%}, "
            f"相对={self.default_weights['relative']:.0%}, "
            f"ESG={self.default_weights['esg']:.0%}, "
            f"情绪={self.default_weights['sentiment']:.0%}"
        )

    @staticmethod
    def _init_industry_weights() -> Dict[str, Dict[str, float]]:
        """
        初始化行业权重模板。

        根据行业特性调整各因子的权重比例：
        - 现金流稳定的行业 → DCF权重更高
        - 可比公司众多的行业 → 相对估值权重更高
        - 高污染/监管行业 → ESG权重更高
        - 高成长/题材行业 → 情绪权重更高

        Returns
        -------
        dict
            行业权重配置
        """
        return {
            # 消费类：现金流稳定，DCF为主
            "食品饮料": {"dcf": 0.45, "relative": 0.25, "esg": 0.15, "sentiment": 0.15},
            "家用电器": {"dcf": 0.40, "relative": 0.25, "esg": 0.15, "sentiment": 0.20},
            "商贸零售": {"dcf": 0.35, "relative": 0.25, "esg": 0.20, "sentiment": 0.20},
            # 金融类：可比性强，相对估值为主
            "银行": {"dcf": 0.20, "relative": 0.45, "esg": 0.20, "sentiment": 0.15},
            "非银金融": {"dcf": 0.20, "relative": 0.40, "esg": 0.20, "sentiment": 0.20},
            "房地产": {"dcf": 0.25, "relative": 0.40, "esg": 0.15, "sentiment": 0.20},
            # 科技类：成长预期，情绪权重高
            "电子": {"dcf": 0.30, "relative": 0.20, "esg": 0.20, "sentiment": 0.30},
            "计算机": {"dcf": 0.25, "relative": 0.20, "esg": 0.20, "sentiment": 0.35},
            "通信": {"dcf": 0.30, "relative": 0.20, "esg": 0.20, "sentiment": 0.30},
            "传媒": {"dcf": 0.25, "relative": 0.20, "esg": 0.25, "sentiment": 0.30},
            # 重资产/高污染：ESG权重大
            "石油石化": {"dcf": 0.30, "relative": 0.15, "esg": 0.40, "sentiment": 0.15},
            "煤炭": {"dcf": 0.30, "relative": 0.15, "esg": 0.40, "sentiment": 0.15},
            "钢铁": {"dcf": 0.30, "relative": 0.15, "esg": 0.40, "sentiment": 0.15},
            "基础化工": {"dcf": 0.30, "relative": 0.20, "esg": 0.35, "sentiment": 0.15},
            "公用事业": {"dcf": 0.40, "relative": 0.20, "esg": 0.25, "sentiment": 0.15},
            # 医药：DCF为主，ESG也重要
            "医药生物": {"dcf": 0.40, "relative": 0.20, "esg": 0.25, "sentiment": 0.15},
        }

    def get_weights(self, industry: Optional[str] = None) -> Dict[str, float]:
        """
        获取指定行业或默认的融合权重。

        Parameters
        ----------
        industry : str, optional
            行业名称，不指定则使用默认权重

        Returns
        -------
        dict
            {"dcf": w1, "relative": w2, "esg": w3, "sentiment": w4}
        """
        if industry and industry in self.industry_weights:
            return self.industry_weights[industry].copy()
        return self.default_weights.copy()

    def set_industry_weights(
        self,
        industry: str,
        dcf: float,
        relative: float,
        esg: float,
        sentiment: float,
    ) -> None:
        """
        手动设置某个行业的融合权重。

        Parameters
        ----------
        industry : str
            行业名称
        dcf : float
            DCF权重
        relative : float
            相对估值权重
        esg : float
            ESG权重
        sentiment : float
            情绪权重
        """
        total = dcf + relative + esg + sentiment
        self.industry_weights[industry] = {
            "dcf": dcf / total,
            "relative": relative / total,
            "esg": esg / total,
            "sentiment": sentiment / total,
        }
        logger.info(f"已设置行业 '{industry}' 的融合权重")

    def fuse(
        self,
        dcf_value: float,
        relative_value: float,
        esg_adjusted_value: float,
        sentiment_adjusted_value: float,
        industry: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        融合四个估值因子得到综合估值。

        Parameters
        ----------
        dcf_value : float
            DCF绝对估值
        relative_value : float
            相对估值（可比公司法）
        esg_adjusted_value : float
            ESG调整后的估值
        sentiment_adjusted_value : float
            情绪校准后的估值
        industry : str, optional
            行业名称（用于选择权重）

        Returns
        -------
        dict
            融合结果，包含各因子贡献和最终估值
        """
        weights = self.get_weights(industry)

        # 加权融合
        contributions = {
            "dcf_contrib": dcf_value * weights["dcf"],
            "relative_contrib": relative_value * weights["relative"],
            "esg_contrib": esg_adjusted_value * weights["esg"],
            "sentiment_contrib": sentiment_adjusted_value * weights["sentiment"],
        }

        final_value = sum(contributions.values())

        # 计算各因子贡献占比
        if final_value != 0:
            contrib_pct = {
                f"{k}_pct": round(v / final_value * 100, 1)
                for k, v in contributions.items()
            }
        else:
            contrib_pct = {f"{k}_pct": 0.0 for k in contributions}

        result = {
            "final_value": round(final_value, 2),
            **{k: round(v, 2) for k, v in contributions.items()},
            **contrib_pct,
            "weights_used": weights,
            "industry": industry or "default",
        }

        return result

    def fuse_with_factor_scores(
        self,
        dcf_value: float,
        relative_value: float,
        esg_score: float,
        sentiment_score: float,
        industry: Optional[str] = None,
        esg_to_value_ratio: float = 0.1,
        sentiment_to_value_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        """
        使用因子分数（而非调整后的估值）进行融合。

        此方法适用于 ESG 和情绪以"分数"而非"价值"形式提供的场景。
        ESG分数自动转换为估值修正系数。

        Parameters
        ----------
        dcf_value : float
            DCF估值
        relative_value : float
            相对估值
        esg_score : float
            ESG综合评分 (0-100)
        sentiment_score : float
            市场情绪综合得分 (-1到1)
        industry : str, optional
            行业
        esg_to_value_ratio : float
            ESG分 → 估值修正系数（每1分对应多少比例修正）
        sentiment_to_value_ratio : float
            情绪分 → 估值修正系数

        Returns
        -------
        dict
            融合结果
        """
        # ESG分数转估值修正
        avg_esg = 50.0
        esg_adjusted = dcf_value * (1 + (esg_score - avg_esg) / 100 * esg_to_value_ratio)

        # 情绪分转估值修正
        sentiment_adjusted = dcf_value * (1 + sentiment_score * sentiment_to_value_ratio)

        return self.fuse(
            dcf_value, relative_value, esg_adjusted, sentiment_adjusted, industry
        )

    def fuse_dataframe(
        self,
        df: pd.DataFrame,
        dcf_col: str = "expected_value",
        relative_col: str = "relative_value",
        esg_col: str = "expected_value_calibrated",
        sentiment_col: str = "expected_value_calibrated",
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        """
        对 DataFrame 进行批量四因子融合。

        Parameters
        ----------
        df : pd.DataFrame
            包含各因子估值的数据表
        dcf_col : str
            DCF估值列名
        relative_col : str
            相对估值列名
        esg_col : str
            ESG调整估值列名
        sentiment_col : str
            情绪校准估值列名
        industry_col : str
            行业列名

        Returns
        -------
        pd.DataFrame
            添加了融合结果列的数据表
        """
        df = df.copy()

        final_values = []
        dcf_contribs = []
        relative_contribs = []
        esg_contribs = []
        sentiment_contribs = []

        for _, row in df.iterrows():
            industry = row.get(industry_col, None)

            result = self.fuse(
                dcf_value=row.get(dcf_col, 0.0),
                relative_value=row.get(relative_col, row.get(dcf_col, 0.0)),
                esg_adjusted_value=row.get(esg_col, row.get(dcf_col, 0.0)),
                sentiment_adjusted_value=row.get(sentiment_col, row.get(dcf_col, 0.0)),
                industry=industry,
            )
            final_values.append(result["final_value"])
            dcf_contribs.append(result["dcf_contrib"])
            relative_contribs.append(result["relative_contrib"])
            esg_contribs.append(result["esg_contrib"])
            sentiment_contribs.append(result["sentiment_contrib"])

        df["fusion_final_value"] = final_values
        df["fusion_dcf_contrib"] = dcf_contribs
        df["fusion_relative_contrib"] = relative_contribs
        df["fusion_esg_contrib"] = esg_contribs
        df["fusion_sentiment_contrib"] = sentiment_contribs

        # 主因子上行空间
        if "current_price" in df.columns:
            df["fusion_upside_pct"] = np.where(
                df["current_price"] > 0,
                ((df["fusion_final_value"] - df["current_price"])
                 / df["current_price"] * 100).round(2),
                0.0,
            )

        logger.info(
            f"批量融合完成: {len(df)} 条记录, "
            f"平均估值={df['fusion_final_value'].mean():.2f}"
        )
        return df


# ============================================================================
# 相对估值计算器
# ============================================================================

class RelativeValuationCalculator:
    """
    相对估值计算器。

    基于行业可比公司的市盈率/市净率/市销率中位数，
    估算目标公司的相对估值。

    Attributes
    ----------
    pe_data : pd.DataFrame
        行业PE数据
    pb_data : pd.DataFrame
        行业PB数据
    """

    def __init__(self) -> None:
        """初始化相对估值计算器。"""
        self.pe_data: pd.DataFrame = pd.DataFrame()
        self.pb_data: pd.DataFrame = pd.DataFrame()
        logger.info("RelativeValuationCalculator 初始化完成")

    def fit(
        self, df: pd.DataFrame, industry_col: str = "industry"
    ) -> "RelativeValuationCalculator":
        """
        从全市场数据计算行业估值中枢。

        Parameters
        ----------
        df : pd.DataFrame
            全市场股票数据（需包含 industry, pe_ttm, pb 列）
        industry_col : str
            行业列名

        Returns
        -------
        self
        """
        for col in ["pe_ttm", "pb"]:
            if col not in df.columns:
                logger.warning(f"列 '{col}' 缺失，相对估值可能不准确")

        # PE中位数
        pe_vals = df.groupby(industry_col)["pe_ttm"].agg(
            median="median", q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75), count="count",
        ).reset_index()
        self.pe_data = pe_vals

        # PB中位数
        if "pb" in df.columns:
            pb_vals = df.groupby(industry_col)["pb"].agg(
                median="median", q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75), count="count",
            ).reset_index()
            self.pb_data = pb_vals

        logger.info(f"相对估值中枢拟合完成: {len(self.pe_data)} 个行业")
        return self

    def estimate(
        self,
        industry: str,
        earnings_per_share: float,
        book_value_per_share: float,
        method: str = "pe",
    ) -> float:
        """
        使用行业中枢估算相对估值。

        Parameters
        ----------
        industry : str
            行业名称
        earnings_per_share : float
            每股收益（EPS）
        book_value_per_share : float
            每股净资产（BPS）
        method : str
            估值方法: "pe" / "pb" / "blended"

        Returns
        -------
        float
            相对估值结果
        """
        pe_val = 0.0
        pb_val = 0.0

        # PE估值
        pe_row = self.pe_data[self.pe_data["industry"] == industry]
        if not pe_row.empty:
            pe_median = pe_row["median"].values[0]
            if pe_median > 0 and pe_median < 200:  # 过滤极端值
                pe_val = earnings_per_share * pe_median

        # PB估值
        if not self.pb_data.empty:
            pb_row = self.pb_data[self.pb_data["industry"] == industry]
            if not pb_row.empty:
                pb_median = pb_row["median"].values[0]
                if pb_median > 0 and pb_median < 50:
                    pb_val = book_value_per_share * pb_median

        if method == "pe":
            return pe_val if pe_val > 0 else pb_val
        elif method == "pb":
            return pb_val if pb_val > 0 else pe_val
        else:  # blended
            if pe_val > 0 and pb_val > 0:
                return pe_val * 0.6 + pb_val * 0.4
            return max(pe_val, pb_val)

    def estimate_dataframe(
        self,
        df: pd.DataFrame,
        industry_col: str = "industry",
        eps_col: str = "eps",
        bps_col: str = "bps",
    ) -> pd.Series:
        """
        批量估算相对估值。

        Parameters
        ----------
        df : pd.DataFrame
            股票数据
        industry_col : str
            行业列
        eps_col : str
            每股收益列
        bps_col : str
            每股净资产列

        Returns
        -------
        pd.Series
            相对估值序列
        """
        return df.apply(
            lambda row: self.estimate(
                row.get(industry_col, ""),
                row.get(eps_col, 0.0),
                row.get(bps_col, row.get("total_equity", 0.0)),
            ),
            axis=1,
        )
