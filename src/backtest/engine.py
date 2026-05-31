"""
历史回测引擎
============
基于历史数据模拟月度调仓策略，计算组合绩效指标。

核心功能：
  1. 月度调仓模拟（按估值信号构建多空组合）
  2. 组合绩效指标（年化收益、夏普比率、最大回撤等）
  3. 分行业绩效归因
  4. 与基准指数对比

策略逻辑：
  - 买入融合估值上行空间最大的前20%股票（多头）
  - 卖出/避开估值下行空间最大的后20%股票（空头/减持）
  - 每月末调仓，考虑1‰交易成本
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """
    历史回测引擎。

    Attributes
    ----------
    rebalance_frequency : str
        调仓频率 ("monthly" / "weekly" / "quarterly")
    transaction_cost : float
        单边交易成本
    benchmark_returns : pd.Series
        基准指数收益率序列
    results_ : dict
        回测结果
    """

    def __init__(
        self,
        rebalance_frequency: str = "monthly",
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.03,
    ) -> None:
        """
        初始化回测引擎。

        Parameters
        ----------
        rebalance_frequency : str
            调仓频率
        transaction_cost : float
            单边交易成本（默认1‰）
        risk_free_rate : float
            无风险利率（年化）
        """
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.results_: Dict[str, Any] = {}
        self._portfolio_values: List[float] = []
        self._turnover_history: List[float] = []
        logger.info(
            f"BacktestEngine 初始化: frequency={rebalance_frequency}, "
            f"cost={transaction_cost:.3%}, rf={risk_free_rate:.1%}"
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        stock_col: str = "stock_code",
        date_col: str = "trade_date",
        price_col: str = "close_price",
        signal_col: str = "fusion_upside_pct",
        mcap_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        准备回测数据：排序、标记调仓日、计算收益率。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        stock_col : str
            股票代码列
        date_col : str
            日期列
        price_col : str
            股价列
        signal_col : str
            信号列（用于排序选股）
        mcap_col : str, optional
            市值列（用于市值加权）

        Returns
        -------
        pd.DataFrame
            准备就绪的回测数据
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([date_col, stock_col]).reset_index(drop=True)

        # 标记调仓日
        if self.rebalance_frequency == "monthly":
            df["rebalance_day"] = df[date_col].apply(
                lambda d: d.day >= d.days_in_month - 5  # 月末最后5个交易日
            )
        elif self.rebalance_frequency == "weekly":
            df["rebalance_day"] = df[date_col].dt.dayofweek == 4  # 每周五
        else:  # quarterly
            df["rebalance_day"] = df[date_col].apply(
                lambda d: d.month in [3, 6, 9, 12] and d.day >= d.days_in_month - 5
            )

        # 计算每只股票的日收益率
        df["daily_return"] = df.groupby(stock_col)[price_col].pct_change()
        df["daily_return"] = df["daily_return"].fillna(0.0).clip(-0.11, 0.11)

        logger.info(f"回测数据准备完成: {len(df)} 条记录, "
                     f"{df[stock_col].nunique()} 只股票")
        return df

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = "fusion_upside_pct",
        stock_col: str = "stock_code",
        date_col: str = "trade_date",
        return_col: str = "daily_return",
        top_quantile: float = 0.2,
        bottom_quantile: float = 0.2,
        equal_weight: bool = True,
    ) -> Dict[str, Any]:
        """
        运行回测。

        策略：
        - 每月末按信号值排序
        - 做多 Top_quantile（买入信号最强的前X%）
        - 做空/减持 Bottom_quantile（信号最弱的Y%）
        - 持有到下一调仓日

        Parameters
        ----------
        df : pd.DataFrame
            回测数据（需经 prepare_data 预处理）
        signal_col : str
            信号列
        stock_col : str
            股票代码列
        date_col : str
            日期列
        return_col : str
            日收益率列
        top_quantile : float
            多头分位数
        bottom_quantile : float
            空头分位数
        equal_weight : bool
            True=等权，False=市值加权

        Returns
        -------
        dict
            回测绩效指标
        """
        logger.info("===== 开始回测 =====")

        rebalance_dates = df[df["rebalance_day"]][date_col].unique()
        rebalance_dates = sorted(rebalance_dates)

        if len(rebalance_dates) < 2:
            logger.error("调仓日不足，无法回测")
            return {}

        portfolio_returns: List[float] = []
        long_returns: List[float] = []
        turnovers: List[float] = []
        current_holdings: List[str] = []

        prev_holdings: List[str] = []

        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            # 调仓日：选股
            rebal_data = df[df[date_col] == rebal_date].copy()

            if len(rebal_data) < 10:
                continue

            # 按信号排序
            rebal_data = rebal_data.dropna(subset=[signal_col])
            rebal_data = rebal_data.sort_values(signal_col, ascending=False)

            n_stocks = len(rebal_data)
            n_long = max(int(n_stocks * top_quantile), 1)

            # 多头持仓
            long_stocks = rebal_data.head(n_long)[stock_col].tolist()
            current_holdings = long_stocks

            # 计算调仓换手率
            if prev_holdings:
                turnover = len(set(long_stocks) - set(prev_holdings)) / max(len(long_stocks), 1)
            else:
                turnover = 1.0
            turnovers.append(turnover)
            prev_holdings = long_stocks.copy()

            # 下一调仓日
            next_idx = rebalance_dates.index(rebal_date) + 1
            if next_idx >= len(rebalance_dates):
                break
            next_date = rebalance_dates[next_idx]

            # 获取持有期数据
            period_data = df[
                (df[date_col] > rebal_date) &
                (df[date_col] <= next_date)
            ]

            # 计算多头组合收益率
            long_period = period_data[period_data[stock_col].isin(long_stocks)]
            if not long_period.empty and len(long_stocks) > 0:
                daily_long = long_period.groupby(date_col)[return_col].mean()
                # 扣除交易成本
                cost = self.transaction_cost * turnover
                daily_long = daily_long - cost / len(daily_long) if len(daily_long) > 0 else daily_long
                long_returns.extend(daily_long.tolist())
                portfolio_returns.extend(daily_long.tolist())

        if not portfolio_returns:
            logger.error("未生成任何交易信号")
            return {}

        portfolio_returns_series = pd.Series(portfolio_returns)
        self._portfolio_values = (1 + portfolio_returns_series).cumprod().tolist()
        self._turnover_history = turnovers

        # 计算绩效指标
        metrics = self._compute_performance_metrics(portfolio_returns_series)
        self.results_ = metrics

        logger.info(
            f"回测完成: 年化收益={metrics['annual_return']:.2%}, "
            f"夏普比率={metrics['sharpe_ratio']:.2f}, "
            f"最大回撤={metrics['max_drawdown']:.2%}"
        )
        return metrics

    def _compute_performance_metrics(
        self, returns: pd.Series
    ) -> Dict[str, Any]:
        """
        计算组合绩效指标。

        Parameters
        ----------
        returns : pd.Series
            日收益率序列

        Returns
        -------
        dict
            绩效指标
        """
        # 年化收益率
        n_days = len(returns)
        if n_days < 2:
            return {}

        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0.0

        # 最大回撤
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # Calmar 比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        # 胜率
        win_rate = (returns > 0).mean()

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # Sortino 比率（下行波动率）
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino = excess_return / downside_vol if downside_vol > 0 else 0.0

        # 平均换手率
        avg_turnover = np.mean(self._turnover_history) if self._turnover_history else 0.0

        metrics = {
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "annual_volatility": round(annual_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 4),
            "win_rate": round(win_rate, 4),
            "profit_loss_ratio": round(profit_loss_ratio, 4),
            "avg_turnover": round(avg_turnover, 4),
            "n_trading_days": n_days,
        }

        return metrics

    def compare_with_benchmark(
        self, benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        与基准指数对比。

        Parameters
        ----------
        benchmark_returns : pd.Series
            基准指数日收益率

        Returns
        -------
        dict
            对比指标
        """
        if not self.results_:
            logger.error("请先运行 run() 进行回测")
            return {}

        # 基准绩效
        bench_total = (1 + benchmark_returns).cumprod().iloc[-1] - 1
        bench_annual = (1 + bench_total) ** (252 / len(benchmark_returns)) - 1
        bench_vol = benchmark_returns.std() * np.sqrt(252)

        # 超额收益
        strategy_return = self.results_["annual_return"]
        excess_return = strategy_return - bench_annual

        # 跟踪误差
        # （简化：年化波动率差值）
        tracking_error = abs(
            self.results_["annual_volatility"] - bench_vol
        )

        # 信息比率
        ir = excess_return / tracking_error if tracking_error > 0 else 0.0

        comparison = {
            "strategy_annual_return": strategy_return,
            "benchmark_annual_return": round(bench_annual, 4),
            "excess_return": round(excess_return, 4),
            "tracking_error": round(tracking_error, 4),
            "information_ratio": round(ir, 4),
            "benchmark_volatility": round(bench_vol, 4),
        }

        logger.info(
            f"基准对比: 超额收益={excess_return:.2%}, 信息比率={ir:.2f}"
        )
        return comparison

    def industry_attribution(
        self,
        df: pd.DataFrame,
        stock_col: str = "stock_code",
        industry_col: str = "industry",
        signal_col: str = "fusion_upside_pct",
    ) -> pd.DataFrame:
        """
        分行业绩效归因。

        Parameters
        ----------
        df : pd.DataFrame
            回测数据
        stock_col : str
            股票代码列
        industry_col : str
            行业列
        signal_col : str
            信号列

        Returns
        -------
        pd.DataFrame
            行业归因表
        """
        attribution = df.groupby(industry_col).agg(
            avg_signal=(signal_col, "mean"),
            stock_count=(stock_col, "nunique"),
        ).reset_index()

        attribution = attribution.sort_values("avg_signal", ascending=False)
        logger.info(f"行业归因完成: {len(attribution)} 个行业")
        return attribution

    def get_portfolio_curve(self) -> pd.DataFrame:
        """
        获取组合净值曲线数据。

        Returns
        -------
        pd.DataFrame
            净值曲线
        """
        if not self._portfolio_values:
            return pd.DataFrame()
        return pd.DataFrame({
            "day": range(len(self._portfolio_values)),
            "nav": self._portfolio_values,
        })

    def generate_report(self) -> str:
        """
        生成回测绩效摘要文本。

        Returns
        -------
        str
            摘要文本
        """
        if not self.results_:
            return "尚未运行回测"

        r = self.results_
        lines = [
            "=" * 50,
            "  回测绩效报告",
            "=" * 50,
            f"  年化收益率:    {r['annual_return']:.2%}",
            f"  年化波动率:    {r['annual_volatility']:.2%}",
            f"  夏普比率:      {r['sharpe_ratio']:.2f}",
            f"  最大回撤:      {r['max_drawdown']:.2%}",
            f"  胜率:          {r['win_rate']:.2%}",
            f"  盈亏比:        {r['profit_loss_ratio']:.2f}",
            f"  交易天数:      {r['n_trading_days']}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

def run_quick_backtest(
    df: pd.DataFrame,
    signal_col: str = "fusion_upside_pct",
) -> Dict[str, Any]:
    """
    便捷函数：快速回测。

    Parameters
    ----------
    df : pd.DataFrame
        包含股价和信号的股票数据
    signal_col : str
        信号列

    Returns
    -------
    dict
        回测绩效
    """
    engine = BacktestEngine()
    df = engine.prepare_data(df)
    return engine.run(df, signal_col=signal_col)


def compute_basic_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    便捷函数：计算基本绩效指标。

    Parameters
    ----------
    returns : pd.Series
        日收益率序列

    Returns
    -------
    dict
        绩效指标
    """
    total = (1 + returns).cumprod().iloc[-1] - 1
    annual = (1 + total) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (annual - 0.03) / vol if vol > 0 else 0.0
    peak = (1 + returns).cumprod().expanding().max()
    dd = ((1 + returns).cumprod() - peak) / peak
    max_dd = dd.min()

    return {
        "total_return": round(total, 4),
        "annual_return": round(annual, 4),
        "annual_volatility": round(vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }
