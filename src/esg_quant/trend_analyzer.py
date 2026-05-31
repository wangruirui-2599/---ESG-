"""
ESG趋势分析模块
===============
对企业的历史ESG评分序列进行时序分析，输出：
  1. 年同比变化率（YoY Change Rate）
  2. 动量分数（基于线性回归斜率的趋势强度）
  3. 趋势分类标签（持续改善/波动/恶化/稳定）

应用场景：
  - 识别ESG持续改善的"进步型企业"（beta因子）
  - 标记ESG恶化的"风险企业"（预警信号）
  - 为动态估值模型提供趋势修正系数
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# ============================================================================
# 趋势分类标签
# ============================================================================

TREND_LABELS = {
    "strong_improve": "显著改善",
    "improve": "改善",
    "stable": "稳定",
    "decline": "恶化",
    "strong_decline": "显著恶化",
    "volatile": "波动",
}

# ============================================================================
# ESG趋势分析器
# ============================================================================

class ESGTrendAnalyzer:
    """
    ESG 时序趋势分析器。

    对每只股票的 ESG 评分时间序列执行：
    - 年变化率计算（YoY）
    - 线性回归趋势检测（动量分数）
    - 多维度趋势分类

    Attributes
    ----------
    min_periods : int
        趋势分析最少需要的观测期数
    momentum_window : int
        动量计算的窗口期（月数）
    significance_level : float
        趋势显著性的p值阈值
    """

    def __init__(
        self,
        min_periods: int = 4,
        momentum_window: int = 12,
        significance_level: float = 0.10,
    ) -> None:
        """
        初始化趋势分析器。

        Parameters
        ----------
        min_periods : int
            最少观测期数（少于此数则标记为"数据不足"）
        momentum_window : int
            动量计算窗口（月），默认12个月
        significance_level : float
            线性回归显著性阈值
        """
        self.min_periods = min_periods
        self.momentum_window = momentum_window
        self.significance_level = significance_level
        logger.info(
            f"ESGTrendAnalyzer 初始化: min_periods={min_periods}, "
            f"momentum_window={momentum_window}, sig_level={significance_level}"
        )

    def compute_yoy_change(
        self,
        df: pd.DataFrame,
        score_col: str = "ESG_total",
        id_col: str = "stock_code",
        date_col: str = "rating_date",
    ) -> pd.DataFrame:
        """
        计算 ESG 评分的年同比变化率。

        YoY Change = (当期值 - 去年同期值) / |去年同期值|

        Parameters
        ----------
        df : pd.DataFrame
            ESG评分数据
        score_col : str
            评分列名
        id_col : str
            股票代码列
        date_col : str
            日期列

        Returns
        -------
        pd.DataFrame
            添加了 yoy_change 和 yoy_direction 列的数据表
        """
        if df.empty:
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([id_col, date_col]).reset_index(drop=True)

        yoy_changes = []
        yoy_directions = []

        for stock, group in df.groupby(id_col):
            group = group.sort_values(date_col)
            scores = group[score_col].values
            n = len(scores)

            # 按季度近似：shift 4 期为去年同期
            lag = min(4, max(1, n // 2))
            group_yoy = [np.nan] * n
            group_dir = [""] * n

            for i in range(lag, n):
                prev = scores[i - lag]
                curr = scores[i]
                if prev != 0:
                    group_yoy[i] = (curr - prev) / abs(prev)
                else:
                    group_yoy[i] = 0.0

                if group_yoy[i] > 0.02:
                    group_dir[i] = "up"
                elif group_yoy[i] < -0.02:
                    group_dir[i] = "down"
                else:
                    group_dir[i] = "flat"

            yoy_changes.extend(group_yoy)
            yoy_directions.extend(group_dir)

        df["yoy_change"] = yoy_changes
        df["yoy_direction"] = yoy_directions

        valid_count = df["yoy_change"].notna().sum()
        avg_change = df["yoy_change"].dropna().mean()
        logger.info(
            f"YoY变化率计算完成: {valid_count}/{len(df)} 有效值, "
            f"均值={avg_change:.4f}"
        )
        return df

    def compute_momentum_score(
        self,
        df: pd.DataFrame,
        score_cols: Optional[List[str]] = None,
        id_col: str = "stock_code",
        date_col: str = "rating_date",
    ) -> pd.DataFrame:
        """
        计算 ESG 动量分数。

        基于最近 momentum_window 个月的数据，对每个ESG维度
        做线性回归，取其斜率（beta）作为动量分数。

        正值 = 改善趋势，负值 = 恶化趋势，|值| = 趋势强度。

        Parameters
        ----------
        df : pd.DataFrame
            ESG评分数据
        score_cols : list of str, optional
            需要分析的评分列，默认为 ESG 四维度
        id_col : str
            股票代码列
        date_col : str
            日期列

        Returns
        -------
        pd.DataFrame
            添加了 momentum_* 列的数据表
        """
        if score_cols is None:
            score_cols = ["E_score", "S_score", "G_score", "ESG_total"]

        # 过滤存在的列
        score_cols = [c for c in score_cols if c in df.columns]
        if not score_cols:
            logger.warning("无有效评分列")
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([id_col, date_col]).reset_index(drop=True)

        for col in score_cols:
            momentum_col = f"{col}_momentum"
            pvalue_col = f"{col}_momentum_pval"
            r2_col = f"{col}_momentum_r2"

            df[momentum_col] = np.nan
            df[pvalue_col] = np.nan
            df[r2_col] = np.nan

            for stock, group in df.groupby(id_col):
                group = group.sort_values(date_col)
                idx = group.index
                scores = group[col].dropna()

                if len(scores) < self.min_periods:
                    continue

                # 取最近 momentum_window 期内数据
                recent = scores.tail(self.momentum_window)
                if len(recent) < self.min_periods:
                    continue

                # 线性回归: score ~ time_index
                x = np.arange(len(recent)).reshape(-1, 1)
                y = recent.values

                try:
                    reg = stats.linregress(x.flatten(), y)
                    # 标准化斜率：除以均值得到相对变化率
                    mean_y = np.mean(y)
                    if mean_y != 0:
                        normalized_slope = reg.slope / abs(mean_y)
                    else:
                        normalized_slope = reg.slope

                    df.loc[group.index[-1], momentum_col] = round(
                        normalized_slope, 6
                    )
                    df.loc[group.index[-1], pvalue_col] = round(reg.pvalue, 4)
                    df.loc[group.index[-1], r2_col] = round(reg.rvalue ** 2, 4)
                except (ValueError, stats.LinAlgError):
                    continue

        valid = df[f"{score_cols[0]}_momentum"].notna().sum()
        logger.info(
            f"动量分数计算完成: {valid}/{df[id_col].nunique()} 只股票有效"
        )
        return df

    def classify_trend(
        self,
        df: pd.DataFrame,
        momentum_col: str = "ESG_total_momentum",
        change_col: str = "yoy_change",
    ) -> pd.DataFrame:
        """
        基于动量分数和YoY变化率对ESG趋势进行分类。

        分类规则（优先度从高到低）：
        - 显著改善: momentum > 0.01 且 YoY > 5% 且 p < 0.05
        - 改善:     momentum > 0.003 且 YoY > 2%
        - 显著恶化: momentum < -0.01 且 YoY < -5% 且 p < 0.05
        - 恶化:     momentum < -0.003 且 YoY < -2%
        - 波动:     |momentum| > 0.005 但 p > 0.10（趋势不显著）
        - 稳定:     其他情况

        Parameters
        ----------
        df : pd.DataFrame
            已计算动量和YoY的数据表
        momentum_col : str
            动量分数列名
        change_col : str
            YoY变化率列名

        Returns
        -------
        pd.DataFrame
            添加了 trend_label, trend_score, trend_confidence 列的数据表
        """
        df = df.copy()

        pvalue_col = momentum_col.replace("_momentum", "_momentum_pval")

        has_pvalue = pvalue_col in df.columns
        has_momentum = momentum_col in df.columns
        has_yoy = change_col in df.columns

        if not has_momentum:
            logger.error(f"动量列 '{momentum_col}' 不存在，请先运行 compute_momentum_score()")
            return df

        labels = []
        scores = []

        for idx, row in df.iterrows():
            mom = row.get(momentum_col, np.nan)
            yoy = row.get(change_col, np.nan) if has_yoy else np.nan
            pval = row.get(pvalue_col, np.nan) if has_pvalue else np.nan

            if pd.isna(mom):
                labels.append("数据不足")
                scores.append(0.0)
                continue

            is_significant = (pval is not None and not pd.isna(pval)
                              and pval < self.significance_level)

            # 显著改善
            if mom > 0.01 and (pd.isna(yoy) or yoy > 0.05) and is_significant:
                labels.append(TREND_LABELS["strong_improve"])
                scores.append(min(mom * 100, 10.0))
            # 改善
            elif mom > 0.003 and (pd.isna(yoy) or yoy > 0.02):
                labels.append(TREND_LABELS["improve"])
                scores.append(min(mom * 100, 5.0))
            # 显著恶化
            elif mom < -0.01 and (pd.isna(yoy) or yoy < -0.05) and is_significant:
                labels.append(TREND_LABELS["strong_decline"])
                scores.append(max(mom * 100, -10.0))
            # 恶化
            elif mom < -0.003 and (pd.isna(yoy) or yoy < -0.02):
                labels.append(TREND_LABELS["decline"])
                scores.append(max(mom * 100, -5.0))
            # 波动（变化大但不显著）
            elif abs(mom) > 0.005 and not is_significant:
                labels.append(TREND_LABELS["volatile"])
                scores.append(mom * 100 * 0.3)
            # 稳定
            else:
                labels.append(TREND_LABELS["stable"])
                scores.append(0.0)

        df["trend_label"] = labels
        df["trend_score"] = [round(s, 4) for s in scores]
        # 置信度 = 1 - p值（p值越小越确信）
        if has_pvalue:
            df["trend_confidence"] = df[pvalue_col].apply(
                lambda p: round(1 - p, 4) if pd.notna(p) else 0.0
            )

        # 统计分类分布
        label_counts = df["trend_label"].value_counts()
        logger.info(f"趋势分类完成: {dict(label_counts)}")
        return df

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        score_cols: Optional[List[str]] = None,
        id_col: str = "stock_code",
        date_col: str = "rating_date",
    ) -> pd.DataFrame:
        """
        一键运行完整 ESG 趋势分析流程。

        Parameters
        ----------
        df : pd.DataFrame
            ESG评分数据
        score_cols : list of str, optional
            评分列
        id_col : str
            股票代码列
        date_col : str
            日期列

        Returns
        -------
        pd.DataFrame
            添加了所有趋势分析列的数据表
        """
        logger.info("===== ESG趋势分析开始 =====")

        # Step 1: YoY变化率
        df = self.compute_yoy_change(df, id_col=id_col, date_col=date_col)
        logger.info("Step 1/3: YoY变化率 ✓")

        # Step 2: 动量分数
        df = self.compute_momentum_score(df, score_cols, id_col, date_col)
        logger.info("Step 2/3: 动量分数 ✓")

        # Step 3: 趋势分类
        df = self.classify_trend(df)
        logger.info("Step 3/3: 趋势分类 ✓")

        logger.info("===== ESG趋势分析完成 =====")
        return df

    def get_trend_summary(
        self, df: pd.DataFrame, id_col: str = "stock_code"
    ) -> pd.DataFrame:
        """
        生成每只股票的ESG趋势摘要。

        Parameters
        ----------
        df : pd.DataFrame
            趋势分析后的数据
        id_col : str
            股票代码列

        Returns
        -------
        pd.DataFrame
            每只股票一行，包含最新趋势指标
        """
        if "trend_label" not in df.columns:
            logger.warning("尚未进行趋势分类，请先运行 classify_trend()")
            return pd.DataFrame()

        # 取每只股票的最新记录
        date_cols = [c for c in df.columns if "date" in c.lower()]
        sort_col = date_cols[0] if date_cols else df.columns[0]

        latest = df.sort_values(sort_col).groupby(id_col).last().reset_index()

        # 选择关键列
        key_cols = [id_col]
        for col in [
            "trend_label", "trend_score", "trend_confidence",
            "yoy_change", "ESG_total_momentum",
            "E_score_momentum", "S_score_momentum", "G_score_momentum",
        ]:
            if col in latest.columns:
                key_cols.append(col)

        summary = latest[key_cols].copy()
        summary = summary.sort_values("trend_score", ascending=False)

        return summary


# ============================================================================
# 便捷函数
# ============================================================================

def analyze_esg_trend(
    df: pd.DataFrame,
    id_col: str = "stock_code",
    date_col: str = "rating_date",
) -> pd.DataFrame:
    """
    便捷函数：一键 ESG 趋势分析。

    Parameters
    ----------
    df : pd.DataFrame
        包含ESG评分的数据
    id_col : str
        股票代码列
    date_col : str
        日期列

    Returns
    -------
    pd.DataFrame
        趋势分析结果
    """
    analyzer = ESGTrendAnalyzer()
    return analyzer.run_full_analysis(df, id_col=id_col, date_col=date_col)


def get_top_improvers(
    df: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """
    获取ESG改善最快的top_n只股票。

    Parameters
    ----------
    df : pd.DataFrame
        趋势分析结果
    top_n : int
        返回数量

    Returns
    -------
    pd.DataFrame
    """
    if "trend_score" not in df.columns:
        analyzer = ESGTrendAnalyzer()
        df = analyzer.run_full_analysis(df)
    return df.nlargest(top_n, "trend_score")
