"""
特征工程模块
============
从原始财务/ESG/市场数据中构建模型就绪的特征。
包含三类特征构建：
  1. 滞后特征 — 捕获历史信息的时序依赖
  2. 滚动统计 — 捕捉趋势和波动性
  3. 行业标准化 — 消除行业差异，使跨行业可比
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 滞后特征构建器
# ============================================================================

class LagFeatureBuilder:
    """
    滞后特征构建器。

    为指定的数值列生成 t-1, t-2, t-3, t-4 等滞后项，
    用于捕捉财务指标的历史趋势和均值回归效应。

    Attributes
    ----------
    lag_periods : list of int
        滞后期数列表
    """

    def __init__(self, lag_periods: Optional[List[int]] = None) -> None:
        """
        初始化滞后特征构建器。

        Parameters
        ----------
        lag_periods : list of int, optional
            滞后期数，默认为 [1, 2, 3, 4]（季度）
        """
        self.lag_periods = lag_periods or [1, 2, 3, 4]
        logger.info(f"LagFeatureBuilder 初始化: 滞后期={self.lag_periods}")

    def build(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "stock_code",
        date_col: str = "report_date",
    ) -> pd.DataFrame:
        """
        为指定列构建滞后特征。

        按股票分组，按日期排序后，对每个特征列生成 lag_k 列。
        例如 ROE 列将生成 ROE_lag_1, ROE_lag_2, ROE_lag_3, ROE_lag_4。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        feature_cols : list of str
            需要构建滞后特征的数值列名
        id_col : str
            股票代码列（用于分组）
        date_col : str
            日期列（用于排序）

        Returns
        -------
        pd.DataFrame
            添加了滞后列的数据表
        """
        if df.empty:
            logger.warning("输入 DataFrame 为空，返回原数据")
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # 确保日期排序
        df = df.sort_values([id_col, date_col]).reset_index(drop=True)

        # 记录原始列顺序
        original_cols = df.columns.tolist()
        new_col_names: List[str] = []

        # 按股票分组生成滞后特征
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"列 '{col}' 不在 DataFrame 中，跳过")
                continue

            for lag in self.lag_periods:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df.groupby(id_col)[col].shift(lag)
                new_col_names.append(lag_col)

        # 统计缺失
        missing_counts = {c: df[c].isna().sum() for c in new_col_names}
        logger.info(
            f"滞后特征构建完成: {len(feature_cols)} 个特征 × "
            f"{len(self.lag_periods)} 个滞后期 = {len(new_col_names)} 个新特征"
        )
        logger.debug(f"滞后特征缺失统计: {missing_counts}")

        return df

    def build_with_diff(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "stock_code",
        date_col: str = "report_date",
    ) -> pd.DataFrame:
        """
        构建滞后特征和差分特征。

        除了 lag_k 列外，额外生成：
        - diff_k: 当期值 - 滞后k期值（绝对变化）
        - pct_change_k: (当期值 - 滞后k期值) / |滞后k期值|（相对变化）

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        feature_cols : list of str
            特征列名
        id_col : str
            股票代码列
        date_col : str
            日期列

        Returns
        -------
        pd.DataFrame
            添加了滞后和差分列的数据表
        """
        df = self.build(df, feature_cols, id_col, date_col)

        for col in feature_cols:
            if col not in df.columns:
                continue
            for lag in self.lag_periods:
                lag_col = f"{col}_lag_{lag}"
                if lag_col in df.columns:
                    # 绝对变化
                    df[f"{col}_diff_{lag}"] = df[col] - df[lag_col]
                    # 相对变化（百分比）
                    df[f"{col}_pct_{lag}"] = (
                        (df[col] - df[lag_col]) / df[lag_col].abs().replace(0, np.nan)
                    )

        logger.info(f"差分特征构建完成")
        return df


# ============================================================================
# 滚动统计特征构建器
# ============================================================================

class RollingStatsBuilder:
    """
    滚动统计特征构建器。

    为每个特征列计算滚动窗口内的聚合统计量，
    包括均值、标准差、最大值、最小值、偏度等。

    Attributes
    ----------
    windows : list of int
        滚动窗口大小列表
    stats : list of str
        需要计算的统计量
    """

    VALID_STATS = {"mean", "std", "min", "max", "median", "skew", "sum"}

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        stats: Optional[List[str]] = None,
    ) -> None:
        """
        初始化滚动统计构建器。

        Parameters
        ----------
        windows : list of int, optional
            滚动窗口大小，默认为 [4, 8, 12]
        stats : list of str, optional
            统计量列表，默认为 ["mean", "std", "min", "max"]
        """
        self.windows = windows or [4, 8, 12]
        self.stats = stats or ["mean", "std", "min", "max"]

        # 验证统计量
        invalid = set(self.stats) - self.VALID_STATS
        if invalid:
            raise ValueError(f"不支持的统计量: {invalid}，可选: {self.VALID_STATS}")

        logger.info(
            f"RollingStatsBuilder 初始化: 窗口={self.windows}, 统计量={self.stats}"
        )

    def build(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "stock_code",
        date_col: str = "report_date",
    ) -> pd.DataFrame:
        """
        构建滚动统计特征。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        feature_cols : list of str
            需要计算滚动统计的数值列
        id_col : str
            股票代码列
        date_col : str
            日期列

        Returns
        -------
        pd.DataFrame
            添加了滚动统计列的数据表
        """
        if df.empty:
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([id_col, date_col]).reset_index(drop=True)

        new_cols: List[str] = []

        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"列 '{col}' 不在 DataFrame 中，跳过")
                continue

            for window in self.windows:
                # 分组滚动对象
                rolled = df.groupby(id_col)[col].rolling(
                    window=window, min_periods=max(1, window // 2)
                )

                for stat in self.stats:
                    new_col = f"{col}_rolling{window}_{stat}"

                    if stat == "mean":
                        df[new_col] = rolled.mean().reset_index(level=0, drop=True)
                    elif stat == "std":
                        df[new_col] = rolled.std().reset_index(level=0, drop=True)
                    elif stat == "min":
                        df[new_col] = rolled.min().reset_index(level=0, drop=True)
                    elif stat == "max":
                        df[new_col] = rolled.max().reset_index(level=0, drop=True)
                    elif stat == "median":
                        df[new_col] = rolled.median().reset_index(level=0, drop=True)
                    elif stat == "skew":
                        df[new_col] = rolled.skew().reset_index(level=0, drop=True)
                    elif stat == "sum":
                        df[new_col] = rolled.sum().reset_index(level=0, drop=True)

                    new_cols.append(new_col)

        logger.info(
            f"滚动统计特征构建完成: {len(feature_cols)} 特征 × "
            f"{len(self.windows)} 窗口 × {len(self.stats)} 统计量 = {len(new_cols)} 个新特征"
        )
        return df

    def build_momentum(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "stock_code",
        date_col: str = "report_date",
    ) -> pd.DataFrame:
        """
        构建动量特征：短期均值 / 长期均值 的比值。

        该比值 >1 表示短期趋势向上，<1 表示向下。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        feature_cols : list of str
            特征列
        id_col : str
            股票代码列
        date_col : str
            日期列

        Returns
        -------
        pd.DataFrame
            添加了动量特征的数据表
        """
        df = self.build(df, feature_cols, id_col, date_col)

        if len(self.windows) >= 2:
            short_w = min(self.windows)
            long_w = max(self.windows)

            for col in feature_cols:
                short_col = f"{col}_rolling{short_w}_mean"
                long_col = f"{col}_rolling{long_w}_mean"
                if short_col in df.columns and long_col in df.columns:
                    # 动量 = 短期均值 / 长期均值
                    df[f"{col}_momentum"] = (
                        df[short_col] / df[long_col].replace(0, np.nan)
                    )
                    # 动量排序分位数
                    df[f"{col}_momentum_rank"] = df.groupby(date_col)[
                        f"{col}_momentum"
                    ].rank(pct=True)

        logger.info("动量特征构建完成")
        return df


# ============================================================================
# 行业标准化器
# ============================================================================

class IndustryStandardizer:
    """
    行业标准化器。

    将财务/ESG指标按其所属行业进行标准化处理，
    消除行业间的系统性差异，使特征在跨行业间可比。

    方法：
    - Z-score 标准化: (x - 行业中位数) / 行业IQR
    - Min-Max 标准化（可选）
    - 行业中位数调整: x - 行业中位数

    Attributes
    ----------
    method : str
        标准化方法
    min_industry_size : int
        行业内最少样本数，不足则使用全市场统计
    """

    def __init__(
        self,
        method: str = "zscore_iqr",
        min_industry_size: int = 5,
    ) -> None:
        """
        初始化行业标准化器。

        Parameters
        ----------
        method : str
            标准化方法:
            - "zscore_iqr": (x - 行业中位数) / 行业IQR
            - "zscore_std": (x - 行业均值) / 行业标准差
            - "median_subtract": x - 行业中位数
            - "minmax": (x - 行业最小值) / (行业最大值 - 行业最小值)
        min_industry_size : int
            行业内最少样本数
        """
        valid_methods = {"zscore_iqr", "zscore_std", "median_subtract", "minmax"}
        if method not in valid_methods:
            raise ValueError(f"不支持的标准化方法: {method}，可选: {valid_methods}")

        self.method = method
        self.min_industry_size = min_industry_size
        self._stats_cache: Dict[str, pd.DataFrame] = {}
        logger.info(
            f"IndustryStandardizer 初始化: 方法={method}, "
            f"最小行业样本={min_industry_size}"
        )

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        """
        拟合并执行行业标准化（一步式）。

        对训练集使用此方法；对测试集请使用 fit() + transform() 两步式。

        Parameters
        ----------
        df : pd.DataFrame
            待标准化数据
        feature_cols : list of str
            需要标准化的数值列
        industry_col : str
            行业列名

        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        stats = self._compute_industry_stats(df, feature_cols, industry_col)
        # 缓存统计量供后续 transform 使用
        self._stats_cache = stats
        return self._apply_transform(df, feature_cols, industry_col, stats)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        industry_col: str = "industry",
    ) -> "IndustryStandardizer":
        """
        计算并缓存行业统计量（训练阶段）。

        Parameters
        ----------
        df : pd.DataFrame
            训练数据
        feature_cols : list of str
            特征列
        industry_col : str
            行业列

        Returns
        -------
        self
        """
        self._stats_cache = self._compute_industry_stats(
            df, feature_cols, industry_col
        )
        logger.info(f"行业统计量已缓存: {len(self._stats_cache)} 个行业")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        """
        使用缓存的统计量进行标准化（推理阶段）。

        Parameters
        ----------
        df : pd.DataFrame
            待标准化数据
        feature_cols : list of str
            特征列
        industry_col : str
            行业列

        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        if not self._stats_cache:
            # 若未 fit，则自动 fit_transform
            logger.warning("未找到缓存统计量，自动执行 fit_transform")
            return self.fit_transform(df, feature_cols, industry_col)
        return self._apply_transform(df, feature_cols, industry_col, self._stats_cache)

    def _compute_industry_stats(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        industry_col: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        计算每个行业 × 每个特征的基本统计量。

        Parameters
        ----------
        df : pd.DataFrame
            数据
        feature_cols : list of str
            特征列
        industry_col : str
            行业列

        Returns
        -------
        dict
            {行业名: DataFrame of stats}
        """
        # 全市场统计（作为小行业回退）
        market_stats = self._compute_stats_frame(df, feature_cols)
        market_stats["industry"] = "__market__"

        stats_dict: Dict[str, pd.DataFrame] = {}
        for industry, group in df.groupby(industry_col):
            if len(group) >= self.min_industry_size:
                stats_dict[industry] = self._compute_stats_frame(group, feature_cols)
                stats_dict[industry]["industry"] = industry
            else:
                # 样本不足，使用全市场统计
                stats_dict[industry] = market_stats.copy()
                stats_dict[industry]["industry"] = industry
                logger.debug(
                    f"行业 '{industry}' 样本不足 ({len(group)} < {self.min_industry_size})，使用全市场统计"
                )

        # 确保市场统计存在
        if "__market__" not in stats_dict:
            stats_dict["__market__"] = market_stats

        return stats_dict

    @staticmethod
    def _compute_stats_frame(
        df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        计算单个分组的基本统计量 DataFrame。

        Parameters
        ----------
        df : pd.DataFrame
            分组数据
        feature_cols : list of str
            特征列

        Returns
        -------
        pd.DataFrame
            统计量（每列一行）
        """
        stats: Dict[str, Dict[str, float]] = {}
        for col in feature_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            stats[col] = {
                "median": series.median() if len(series) > 0 else 0.0,
                "q1": series.quantile(0.25) if len(series) > 0 else 0.0,
                "q3": series.quantile(0.75) if len(series) > 0 else 1.0,
                "mean": series.mean() if len(series) > 0 else 0.0,
                "std": series.std() if len(series) > 0 else 1.0,
                "min": series.min() if len(series) > 0 else 0.0,
                "max": series.max() if len(series) > 0 else 1.0,
            }
        return pd.DataFrame(stats).T

    def _apply_transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        industry_col: str,
        stats_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        应用标准化变换。

        Parameters
        ----------
        df : pd.DataFrame
            数据
        feature_cols : list of str
            特征列
        industry_col : str
            行业列
        stats_dict : dict
            统计量字典

        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        df = df.copy()
        changed_cols: List[str] = []

        for col in feature_cols:
            if col not in df.columns:
                continue

            new_col = f"{col}_ind_std"
            df[new_col] = df[col].astype(float).copy()
            changed_cols.append(new_col)

            for industry in df[industry_col].unique():
                mask = df[industry_col] == industry
                if not mask.any():
                    continue

                # 获取行业统计，不存在则回退到全市场
                industry_stats = stats_dict.get(
                    industry, stats_dict.get("__market__")
                )
                if industry_stats is None or col not in industry_stats.index:
                    continue

                row = industry_stats.loc[col]
                values = df.loc[mask, col].astype(float)

                if self.method == "zscore_iqr":
                    iqr = max(row["q3"] - row["q1"], 1e-8)
                    df.loc[mask, new_col] = (values - row["median"]) / iqr

                elif self.method == "zscore_std":
                    std = max(row["std"], 1e-8)
                    df.loc[mask, new_col] = (values - row["mean"]) / std

                elif self.method == "median_subtract":
                    df.loc[mask, new_col] = values - row["median"]

                elif self.method == "minmax":
                    range_val = max(row["max"] - row["min"], 1e-8)
                    df.loc[mask, new_col] = (values - row["min"]) / range_val

        logger.info(
            f"行业标准化完成: {len(changed_cols)} 个特征, "
            f"方法={self.method}"
        )
        return df


# ============================================================================
# 特征工程管线
# ============================================================================

class FeatureEngineeringPipeline:
    """
    特征工程管线。

    将滞后、滚动和标准化三个构建器串联，
    一键完成从原始数据到模型就绪特征的全流程。

    Attributes
    ----------
    lag_builder : LagFeatureBuilder
    rolling_builder : RollingStatsBuilder
    standardizer : IndustryStandardizer
    """

    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        standardize_method: str = "zscore_iqr",
        min_industry_size: int = 5,
    ) -> None:
        """
        初始化特征工程管线。

        Parameters
        ----------
        lag_periods : list of int, optional
            滞后期数
        rolling_windows : list of int, optional
            滚动窗口
        standardize_method : str
            标准化方法
        min_industry_size : int
            最小行业样本数
        """
        self.lag_builder = LagFeatureBuilder(lag_periods)
        self.rolling_builder = RollingStatsBuilder(rolling_windows)
        self.standardizer = IndustryStandardizer(
            method=standardize_method,
            min_industry_size=min_industry_size,
        )
        logger.info("FeatureEngineeringPipeline 初始化完成")

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "stock_code",
        date_col: str = "report_date",
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        """
        运行完整特征工程管线。

        流程：
        1. 构建滞后特征和差分特征
        2. 构建滚动统计和动量特征
        3. 对所有原始特征列进行行业标准化

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        feature_cols : list of str
            需要处理的数值特征列
        id_col : str
            股票标识列
        date_col : str
            日期列
        industry_col : str
            行业列

        Returns
        -------
        pd.DataFrame
            添加了所有工程特征的数据表
        """
        n_cols_before = len(df.columns)

        # Step 1: 滞后特征
        df = self.lag_builder.build_with_diff(df, feature_cols, id_col, date_col)
        n_after_lag = len(df.columns)
        logger.info(f"Step 1/3 滞后特征: +{n_after_lag - n_cols_before} 列")

        # Step 2: 滚动统计
        df = self.rolling_builder.build_momentum(df, feature_cols, id_col, date_col)
        n_after_rolling = len(df.columns)
        logger.info(f"Step 2/3 滚动统计: +{n_after_rolling - n_after_lag} 列")

        # Step 3: 行业标准化
        df = self.standardizer.fit_transform(df, feature_cols, industry_col)
        n_after_std = len(df.columns)
        logger.info(f"Step 3/3 行业标准化: +{n_after_std - n_after_rolling} 列")

        logger.info(
            f"特征工程管线完成: {n_cols_before} → {n_after_std} 列 "
            f"(+{n_after_std - n_cols_before} 个新特征)"
        )
        return df


# ============================================================================
# 便捷函数
# ============================================================================

def create_lag_features(
    df: pd.DataFrame,
    cols: List[str],
    lags: Optional[List[int]] = None,
    stock_col: str = "stock_code",
    date_col: str = "report_date",
) -> pd.DataFrame:
    """
    便捷函数：快速创建滞后特征。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    cols : list of str
        特征列
    lags : list of int, optional
        滞后期数
    stock_col : str
        股票代码列
    date_col : str
        日期列

    Returns
    -------
    pd.DataFrame
    """
    builder = LagFeatureBuilder(lags)
    return builder.build(df, cols, stock_col, date_col)


def create_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: Optional[List[int]] = None,
    stock_col: str = "stock_code",
    date_col: str = "report_date",
) -> pd.DataFrame:
    """
    便捷函数：快速创建滚动统计特征。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    cols : list of str
        特征列
    windows : list of int, optional
        滚动窗口
    stock_col : str
        股票代码列
    date_col : str
        日期列

    Returns
    -------
    pd.DataFrame
    """
    builder = RollingStatsBuilder(windows)
    return builder.build(df, cols, stock_col, date_col)


def standardize_by_industry(
    df: pd.DataFrame,
    cols: List[str],
    industry_col: str = "industry",
    method: str = "zscore_iqr",
) -> pd.DataFrame:
    """
    便捷函数：快速执行行业标准化。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    cols : list of str
        特征列
    industry_col : str
        行业列
    method : str
        标准化方法

    Returns
    -------
    pd.DataFrame
    """
    std = IndustryStandardizer(method=method)
    return std.fit_transform(df, cols, industry_col)
