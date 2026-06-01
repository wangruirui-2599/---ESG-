"""
因果推断模块
============
使用双重差分法（Difference-in-Differences, DID）评估
ESG评级变化对企业估值和市场表现的影响。

DID 模型：
  Y_it = β₀ + β₁·Treat_i + β₂·Post_t + β₃·(Treat_i × Post_t) + γ·Controls + ε_it

其中：
  - Treat_i = 1 if ESG评级上调（处理组），0 if ESG评级不变（对照组）
  - Post_t   = 1 if 评级调整之后，0 if 之前
  - β₃       = 因果效应（DID估计量）

应用场景：
  - 评估ESG评级上调是否带来估值提升
  - 评估ESG负面事件是否导致估值折价
  - 控制内生性后的净效应估计
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
import statsmodels.api as sm
from statsmodels.iolib.summary import Summary


# ============================================================================
# DID 分析器
# ============================================================================

class DIDAnalyzer:
    """
    双重差分法 (DID) 因果推断分析器。

    Attributes
    ----------
    results_ : dict
        DID 回归结果
    parallel_trend_test_ : dict
        平行趋势检验结果
    """

    def __init__(self) -> None:
        """初始化 DID 分析器。"""
        self.results_: Dict[str, Any] = {}
        self.parallel_trend_test_: Dict[str, Any] = {}
        logger.info("DIDAnalyzer 初始化完成")

    def prepare_did_data(
        self,
        df: pd.DataFrame,
        event_date: str,
        pre_window: int = 4,
        post_window: int = 4,
        stock_col: str = "stock_code",
        date_col: str = "trade_date",
        outcome_col: str = "close_price",
        treat_condition: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        构建 DID 分析所需的面板数据。

        将数据划分为事件前后的处理组和对照组。

        Parameters
        ----------
        df : pd.DataFrame
            面板数据
        event_date : str
            事件发生日期（如ESG评级调整日）
        pre_window : int
            事件前窗口期（季度数）
        post_window : int
            事件后窗口期
        stock_col : str
            股票代码列
        date_col : str
            日期列
        outcome_col : str
            结果变量列
        treat_condition : pd.Series, optional
            自定义处理组标记（True=处理组），不指定则自动识别

        Returns
        -------
        pd.DataFrame
            DID分析数据，包含 Treat, Post, Treat×Post 等列
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        event = pd.to_datetime(event_date)

        # 时间窗口筛选
        pre_start = event - pd.DateOffset(months=pre_window * 3)
        post_end = event + pd.DateOffset(months=post_window * 3)

        window_df = df[
            (df[date_col] >= pre_start) &
            (df[date_col] <= post_end)
        ].copy()

        # 标记 Post 时期
        window_df["Post"] = (window_df[date_col] >= event).astype(int)

        # 标记处理组
        if treat_condition is not None:
            window_df["Treat"] = treat_condition.loc[
                treat_condition.index.intersection(window_df.index)
            ].astype(int)
        else:
            # 自动识别：ESG评级变化>0的为处理组
            if "ESG_total" in window_df.columns and "ESG_total_lag_4" in window_df.columns:
                window_df["esg_change"] = (
                    window_df["ESG_total"] - window_df["ESG_total_lag_4"]
                )
                window_df["Treat"] = (window_df["esg_change"] > 5).astype(int)
            else:
                logger.warning("无法自动识别处理组，请提供 treat_condition")
                window_df["Treat"] = 0

        # 交互项 DID = Treat × Post
        window_df["DID"] = window_df["Treat"] * window_df["Post"]

        n_treat = window_df["Treat"].sum()
        n_control = len(window_df) - n_treat
        logger.info(
            f"DID数据准备完成: {len(window_df)} 条观测, "
            f"处理组={n_treat}, 对照组={n_control}"
        )
        return window_df

    def run_did(
        self,
        did_data: pd.DataFrame,
        outcome_col: str = "close_price",
        control_cols: Optional[List[str]] = None,
        robust_se: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 DID 回归估计。

        Y = β₀ + β₁·Treat + β₂·Post + β₃·(Treat×Post) + γ·Controls + ε

        Parameters
        ----------
        did_data : pd.DataFrame
            prepare_did_data 的输出
        outcome_col : str
            被解释变量
        control_cols : list of str, optional
            控制变量列表
        robust_se : bool
            是否使用异方差稳健标准误

        Returns
        -------
        dict
            回归结果
        """
        logger.info("===== 执行 DID 回归 =====")

        # 构建回归变量
        X_cols = ["Treat", "Post", "DID"]
        if control_cols:
            X_cols += [c for c in control_cols if c in did_data.columns]

        # 处理缺失值
        model_data = did_data[[outcome_col] + X_cols].dropna()

        if len(model_data) < 30:
            logger.error(f"有效样本不足: {len(model_data)}")
            return {}

        X = model_data[X_cols]
        X = sm.add_constant(X)  # 添加截距
        y = model_data[outcome_col]

        # OLS 回归
        if robust_se:
            model = sm.OLS(y, X).fit(
                cov_type="HC1",  # 异方差稳健标准误
                use_t=True,
            )
        else:
            model = sm.OLS(y, X).fit()

        # 提取 DID 系数
        did_coef = model.params.get("DID", np.nan)
        did_pvalue = model.pvalues.get("DID", np.nan)
        did_se = model.bse.get("DID", np.nan)

        # 效应量（Cohen's d 近似）
        y_std = y.std()
        effect_size = did_coef / y_std if y_std > 0 else 0.0

        # 显著性判断
        if did_pvalue < 0.01:
            significance = "***"
        elif did_pvalue < 0.05:
            significance = "**"
        elif did_pvalue < 0.10:
            significance = "*"
        else:
            significance = "不显著"

        self.results_ = {
            "did_coefficient": round(did_coef, 6),
            "did_std_error": round(did_se, 6),
            "did_p_value": round(did_pvalue, 6),
            "did_t_statistic": round(did_coef / did_se, 4) if did_se > 0 else 0.0,
            "effect_size_cohens_d": round(effect_size, 4),
            "significance": significance,
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "n_observations": int(model.nobs),
            "all_coefficients": model.params.to_dict(),
            "all_pvalues": model.pvalues.to_dict(),
        }

        logger.info(
            f"DID结果: β₃(DID)={did_coef:.4f}, "
            f"p={did_pvalue:.4f} {significance}, "
            f"R²={model.rsquared:.3f}, N={model.nobs}"
        )
        return self.results_

    def test_parallel_trends(
        self,
        df: pd.DataFrame,
        outcome_col: str = "close_price",
        date_col: str = "trade_date",
        event_date: str = "",
        stock_col: str = "stock_code",
        treat_col: str = "Treat",
        pre_periods: int = 4,
    ) -> Dict[str, Any]:
        """
        检验平行趋势假设（DID 的关键前提）。

        使用事件研究法：在事件前各期，
        处理组和对照组的趋势应无显著差异。

        Parameters
        ----------
        df : pd.DataFrame
            面板数据
        outcome_col : str
            结果变量
        date_col : str
            日期列
        event_date : str
            事件日期
        stock_col : str
            股票代码列
        treat_col : str
            处理组标记列
        pre_periods : int
            事件前检验期数

        Returns
        -------
        dict
            平行趋势检验结果
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        event = pd.to_datetime(event_date) if event_date else pd.Timestamp("2024-01-01")

        # 按季度分组
        df["quarter"] = df[date_col].dt.to_period("Q")
        df["quarters_to_event"] = (
            (df[date_col] - event).dt.days / 90.0
        ).round().astype(int)

        # 事件前各期的组间差异检验
        test_results = []
        for period in range(-pre_periods, 0):
            period_data = df[df["quarters_to_event"] == period]
            if len(period_data) < 10:
                continue

            treat_group = period_data[period_data[treat_col] == 1][outcome_col]
            control_group = period_data[period_data[treat_col] == 0][outcome_col]

            if len(treat_group) < 3 or len(control_group) < 3:
                continue

            t_stat, p_value = stats.ttest_ind(treat_group, control_group)
            test_results.append({
                "period": period,
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
                "treat_mean": round(treat_group.mean(), 4),
                "control_mean": round(control_group.mean(), 4),
                "diff": round(treat_group.mean() - control_group.mean(), 4),
            })

        # 判断平行趋势成立与否
        sig_count = sum(1 for r in test_results if r["significant"])
        parallel_holds = sig_count <= 1  # 允许至多1期显著

        self.parallel_trend_test_ = {
            "parallel_trends_holds": parallel_holds,
            "significant_periods": sig_count,
            "total_periods_tested": len(test_results),
            "period_details": test_results,
            "verdict": "✓ 平行趋势成立" if parallel_holds else "⚠ 平行趋势可能不成立",
        }

        logger.info(
            f"平行趋势检验: {self.parallel_trend_test_['verdict']} "
            f"({sig_count}/{len(test_results)} 期显著)"
        )
        return self.parallel_trend_test_

    def placebo_test(
        self,
        did_data: pd.DataFrame,
        outcome_col: str = "close_price",
        n_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        安慰剂检验：随机打乱处理组分配，观察DID系数分布。

        如果真实DID系数显著偏离随机分布的中心，
        则证明因果效应不是偶然的。

        Parameters
        ----------
        did_data : pd.DataFrame
            DID数据
        outcome_col : str
            结果变量
        n_iterations : int
            模拟次数

        Returns
        -------
        dict
            安慰剂检验结果
        """
        real_did = self.results_.get("did_coefficient", 0.0)
        placebo_coeffs = []

        for i in range(n_iterations):
            # 随机打乱 Treat 标记
            shuffled = did_data.copy()
            shuffled["Treat"] = np.random.permutation(shuffled["Treat"].values)
            shuffled["DID"] = shuffled["Treat"] * shuffled["Post"]

            X = sm.add_constant(shuffled[["Treat", "Post", "DID"]])
            y = shuffled[outcome_col].dropna()
            X = X.loc[y.index]

            try:
                model = sm.OLS(y, X).fit()
                placebo_coeffs.append(model.params.get("DID", 0.0))
            except Exception:
                continue

        placebo_coeffs = np.array(placebo_coeffs)
        p_value_placebo = (np.abs(placebo_coeffs) >= np.abs(real_did)).mean()

        result = {
            "real_did_coefficient": round(real_did, 6),
            "placebo_mean": round(placebo_coeffs.mean(), 6),
            "placebo_std": round(placebo_coeffs.std(), 6),
            "placebo_p_value": round(p_value_placebo, 4),
            "n_placebo_iterations": len(placebo_coeffs),
            "significant": p_value_placebo < 0.05,
            "percentile_95": round(np.percentile(np.abs(placebo_coeffs), 95), 6),
        }

        logger.info(
            f"安慰剂检验: 真实DID={real_did:.4f}, "
            f"安慰剂均值={placebo_coeffs.mean():.4f}, "
            f"p={p_value_placebo:.4f}"
        )
        return result

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        event_date: str,
        outcome_col: str = "close_price",
        control_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        一键运行完整 DID 分析流程。

        Parameters
        ----------
        df : pd.DataFrame
            面板数据
        event_date : str
            事件日期
        outcome_col : str
            结果变量
        control_cols : list of str, optional
            控制变量

        Returns
        -------
        dict
            完整分析结果
        """
        logger.info("===== DID 因果推断开始 =====")

        # 1. 准备数据
        did_data = self.prepare_did_data(df, event_date=event_date,
                                          outcome_col=outcome_col)

        # 2. 平行趋势检验
        trend_test = self.test_parallel_trends(
            did_data, outcome_col=outcome_col, event_date=event_date
        )
        logger.info(f"Step 1/3: 平行趋势检验 → {trend_test['verdict']}")

        # 3. DID回归
        did_results = self.run_did(did_data, outcome_col=outcome_col,
                                   control_cols=control_cols)
        logger.info(f"Step 2/3: DID回归 R²={did_results.get('r_squared', 0):.3f}")

        # 4. 安慰剂检验
        placebo = self.placebo_test(did_data, outcome_col=outcome_col)
        logger.info(f"Step 3/3: 安慰剂检验 p={placebo['placebo_p_value']:.4f}")

        full_results = {
            "did_results": did_results,
            "parallel_trends": trend_test,
            "placebo_test": placebo,
            "conclusion": self._generate_conclusion(did_results, trend_test, placebo),
        }

        logger.info(f"===== DID分析完成: {full_results['conclusion']} =====")
        return full_results

    @staticmethod
    def _generate_conclusion(
        did_results: Dict[str, Any],
        trend_test: Dict[str, Any],
        placebo: Dict[str, Any],
    ) -> str:
        """生成 DID 分析的总结论。"""
        did_valid = did_results.get("significance", "不显著") in ("***", "**", "*")
        trend_ok = trend_test.get("parallel_trends_holds", False)
        placebo_ok = placebo.get("significant", False)

        if did_valid and trend_ok and placebo_ok:
            return "ESG评级变化对估值有显著的因果效应（通过所有检验）"
        elif did_valid and trend_ok:
            return "ESG评级变化可能对估值有影响（DID显著+平行趋势成立，安慰剂检验未通过）"
        elif did_valid:
            return "ESG评级变化与估值变化相关，但因果推断需谨慎（平行趋势可能不成立）"
        else:
            return "未发现ESG评级变化对估值的显著因果效应"


# ============================================================================
# 便捷函数
# ============================================================================

def run_did_analysis(
    df: pd.DataFrame,
    event_date: str,
    outcome_col: str = "close_price",
) -> Dict[str, Any]:
    """
    便捷函数：快速 DID 因果推断。

    Parameters
    ----------
    df : pd.DataFrame
        面板数据
    event_date : str
        事件日期
    outcome_col : str
        结果变量

    Returns
    -------
    dict
        DID分析结果
    """
    analyzer = DIDAnalyzer()
    return analyzer.run_full_analysis(df, event_date=event_date,
                                       outcome_col=outcome_col)


def event_study(
    df: pd.DataFrame,
    event_dates: Dict[str, str],
    outcome_col: str = "close_price",
    pre_window: int = 4,
    post_window: int = 4,
) -> pd.DataFrame:
    """
    便捷函数：对多个事件批量进行事件研究（简化版DID）。

    Parameters
    ----------
    df : pd.DataFrame
        面板数据
    event_dates : dict
        {股票代码: 事件日期}
    outcome_col : str
        结果变量
    pre_window : int
        事件前窗口
    post_window : int
        事件后窗口

    Returns
    -------
    pd.DataFrame
        各事件的因果效应汇总
    """
    analyzer = DIDAnalyzer()
    results = []
    for stock, date in event_dates.items():
        stock_data = df[df["stock_code"] == stock]
        if len(stock_data) < pre_window + post_window:
            continue
        try:
            did_data = analyzer.prepare_did_data(
                stock_data, event_date=date, pre_window=pre_window,
                post_window=post_window, outcome_col=outcome_col,
            )
            did_result = analyzer.run_did(did_data, outcome_col=outcome_col)
            did_result["stock_code"] = stock
            did_result["event_date"] = date
            results.append(did_result)
        except Exception as e:
            logger.error(f"事件研究失败 [{stock}]: {e}")

    return pd.DataFrame(results)
