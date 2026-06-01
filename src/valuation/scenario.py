"""
多情景DCF估值引擎
==================
基于自由现金流折现模型（DCF），在乐观/中性/悲观三种情景下
对企业进行估值，并通过概率加权输出期望估值。

公式：
  DCF = Σ(FCF_t / (1+WACC)^t) + Terminal_Value / (1+WACC)^n

情景概率由 probabilistic.py 中的决策模型动态分配。
默认使用配置文件中定义的先验概率。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 情景参数数据类
# ============================================================================

@dataclass
class ScenarioParams:
    """单个估值情景的参数集"""
    name: str                                    # 情景名称
    revenue_growth: float                        # 营收复合增长率
    margin_change: float                         # 净利润率变化（百分点）
    wacc: float                                  # 加权平均资本成本
    terminal_growth: float                       # 永续增长率
    esg_premium: float = 0.0                     # ESG溢价/折价比例
    probability: float = 0.33                    # 情景先验概率


@dataclass
class DCFOutput:
    """DCF估值输出"""
    stock_code: str
    scenario_name: str
    intrinsic_value: float                       # 每股内在价值
    current_price: float                         # 当前股价
    upside_pct: float                            # 上行空间百分比
    fcf_projections: List[float] = field(default_factory=list)
    terminal_value: float = 0.0
    present_value_fcf: float = 0.0
    present_value_terminal: float = 0.0


# ============================================================================
# DCF估值引擎
# ============================================================================

class DCFValuator:
    """
    多情景 DCF 估值器。

    对企业未来自由现金流进行折现，计算每股内在价值。

    Attributes
    ----------
    forecast_years : int
        显式预测期年数
    scenarios : list of ScenarioParams
        估值情景列表
    """

    def __init__(
        self,
        forecast_years: int = 5,
        scenarios: Optional[List[ScenarioParams]] = None,
    ) -> None:
        """
        初始化DCF估值器。

        Parameters
        ----------
        forecast_years : int
            显式预测期年数（默认5年）
        scenarios : list of ScenarioParams, optional
            估值情景列表，默认使用内置三情景
        """
        self.forecast_years = forecast_years
        self.scenarios = scenarios or self._default_scenarios()
        logger.info(
            f"DCFValuator 初始化: {forecast_years}年预测期, "
            f"{len(self.scenarios)} 个情景"
        )

    @staticmethod
    def _default_scenarios() -> List[ScenarioParams]:
        """构建默认三情景参数。"""
        return [
            ScenarioParams("乐观", 0.15, 0.03, 0.08, 0.035, 0.10, 0.25),
            ScenarioParams("中性", 0.08, 0.00, 0.10, 0.025, 0.00, 0.50),
            ScenarioParams("悲观", 0.02, -0.05, 0.12, 0.015, -0.15, 0.25),
        ]

    def load_scenarios(self, config_data: List[Dict[str, Any]]) -> None:
        """
        从配置加载情景参数。

        Parameters
        ----------
        config_data : list of dict
            情景参数配置列表
        """
        self.scenarios = []
        for item in config_data:
            self.scenarios.append(ScenarioParams(
                name=item.get("name", ""),
                revenue_growth=item.get("revenue_growth", 0.08),
                margin_change=item.get("margin_change", 0.0),
                wacc=item.get("wacc", 0.10),
                terminal_growth=item.get("terminal_growth", 0.025),
                esg_premium=item.get("esg_premium", 0.0),
                probability=item.get("probability", 0.33),
            ))
        logger.info(f"情景参数加载完成: {len(self.scenarios)} 个情景")

    def _project_fcf(
        self,
        base_revenue: float,
        base_margin: float,
        growth_rate: float,
        margin_change: float,
        capex_ratio: float = 0.08,
        deprec_ratio: float = 0.05,
    ) -> List[float]:
        """
        预测未来 N 年的自由现金流。

        FCF = EBIT × (1 - tax_rate) + 折旧 - 资本支出 - 营运资本变动
        简化版：FCF = Revenue × Margin × (1 - tax) × (1 - Capex/Revenue)
                    + Revenue × Deprec/Revenue

        Parameters
        ----------
        base_revenue : float
            基准年营收
        base_margin : float
            基准年净利润率
        growth_rate : float
            复合增长率
        margin_change : float
            利润率逐年变化量（百分点）
        capex_ratio : float
            资本支出占营收比例
        deprec_ratio : float
            折旧占营收比例

        Returns
        -------
        list of float
            未来每年的FCF预测值
        """
        fcf_list = []
        tax_rate = 0.25  # 企业所得税率25%

        for t in range(1, self.forecast_years + 1):
            revenue = base_revenue * (1 + growth_rate) ** t
            margin = base_margin + margin_change * t
            # 限制利润率在合理范围
            margin = np.clip(margin, -0.30, 0.60)

            ebit = revenue * margin
            nopat = ebit * (1 - tax_rate)
            capex = revenue * capex_ratio
            depreciation = revenue * deprec_ratio
            # 简化：营运资本变动 = 营收增长的5%
            wc_change = base_revenue * growth_rate * 0.05

            fcf = nopat + depreciation - capex - wc_change
            fcf_list.append(max(fcf, 0.0))  # FCF不低于0

        return fcf_list

    def _calculate_terminal_value(
        self,
        last_fcf: float,
        wacc: float,
        terminal_growth: float,
    ) -> float:
        """
        计算终值（Gordon Growth Model）。

        TV = FCF_n × (1 + g) / (WACC - g)

        Parameters
        ----------
        last_fcf : float
            最后一年的FCF
        wacc : float
            折现率
        terminal_growth : float
            永续增长率

        Returns
        -------
        float
            终值
        """
        if wacc <= terminal_growth:
            logger.warning(f"WACC({wacc:.3f}) <= g({terminal_growth:.3f})，使用g=WACC-0.01")
            terminal_growth = wacc - 0.01
        return last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

    def value_single_scenario(
        self,
        stock_code: str,
        base_revenue: float,
        base_margin: float,
        current_price: float,
        total_shares: float,
        scenario: ScenarioParams,
        net_debt: float = 0.0,
    ) -> DCFOutput:
        """
        在单个情景下对一只股票进行DCF估值。

        Parameters
        ----------
        stock_code : str
            股票代码
        base_revenue : float
            基准年营收（亿元）
        base_margin : float
            基准年净利润率
        current_price : float
            当前股价
        total_shares : float
            总股本（亿股）
        scenario : ScenarioParams
            情景参数
        net_debt : float
            净债务（亿元，总债务 - 现金）

        Returns
        -------
        DCFOutput
            估值结果
        """
        # 预测FCF
        fcf_list = self._project_fcf(
            base_revenue, base_margin,
            scenario.revenue_growth,
            scenario.margin_change,
        )

        # DCF折现
        pv_fcf = 0.0
        for t, fcf in enumerate(fcf_list, start=1):
            pv_fcf += fcf / (1 + scenario.wacc) ** t

        # 终值折现
        terminal_value = self._calculate_terminal_value(
            fcf_list[-1] if fcf_list else base_revenue * base_margin,
            scenario.wacc,
            scenario.terminal_growth,
        )
        pv_terminal = terminal_value / (1 + scenario.wacc) ** self.forecast_years

        # 企业价值 → 股权价值 → 每股价值
        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value - net_debt

        if total_shares > 0:
            intrinsic_per_share = equity_value / total_shares
        else:
            intrinsic_per_share = 0.0

        # ESG溢价/折价调整
        intrinsic_per_share *= (1 + scenario.esg_premium)

        # 上行空间
        if current_price > 0:
            upside = (intrinsic_per_share - current_price) / current_price
        else:
            upside = 0.0

        return DCFOutput(
            stock_code=stock_code,
            scenario_name=scenario.name,
            intrinsic_value=round(intrinsic_per_share, 2),
            current_price=round(current_price, 2),
            upside_pct=round(upside * 100, 2),
            fcf_projections=[round(f, 2) for f in fcf_list],
            terminal_value=round(terminal_value, 2),
            present_value_fcf=round(pv_fcf, 2),
            present_value_terminal=round(pv_terminal, 2),
        )

    def value_all_scenarios(
        self,
        stock_code: str,
        base_revenue: float,
        base_margin: float,
        current_price: float,
        total_shares: float,
        net_debt: float = 0.0,
    ) -> List[DCFOutput]:
        """
        在所有情景下对一只股票进行估值。

        Parameters
        ----------
        stock_code : str
            股票代码
        base_revenue : float
            基准营收
        base_margin : float
            基准利润率
        current_price : float
            当前股价
        total_shares : float
            总股本
        net_debt : float
            净债务

        Returns
        -------
        list of DCFOutput
            各情景的估值结果
        """
        results = []
        for scenario in self.scenarios:
            result = self.value_single_scenario(
                stock_code, base_revenue, base_margin,
                current_price, total_shares,
                scenario, net_debt,
            )
            results.append(result)

        return results

    def value_expected(
        self,
        stock_code: str,
        base_revenue: float,
        base_margin: float,
        current_price: float,
        total_shares: float,
        net_debt: float = 0.0,
        scenario_probabilities: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        概率加权期望估值。

        使用各情景的先验概率（或用户指定的概率）对估值加权平均。

        Parameters
        ----------
        stock_code : str
            股票代码
        base_revenue : float
            基准营收
        base_margin : float
            基准利润率
        current_price : float
            当前股价
        total_shares : float
            总股本
        net_debt : float
            净债务
        scenario_probabilities : list of float, optional
            自定义情景概率，不指定则使用情景自带的先验概率

        Returns
        -------
        dict
            包含期望估值、情景明细、上下行风险的字典
        """
        results = self.value_all_scenarios(
            stock_code, base_revenue, base_margin,
            current_price, total_shares, net_debt,
        )

        if scenario_probabilities is None:
            scenario_probabilities = [s.probability for s in self.scenarios]

        # 归一化概率
        total_prob = sum(scenario_probabilities)
        if total_prob > 0:
            scenario_probabilities = [p / total_prob for p in scenario_probabilities]
        else:
            scenario_probabilities = [1.0 / len(self.scenarios)] * len(self.scenarios)

        # 加权期望估值
        expected_value = sum(
            r.intrinsic_value * p
            for r, p in zip(results, scenario_probabilities)
        )

        # 上下行风险（基于乐观/悲观情景）
        optimistic = results[0].intrinsic_value if results else 0
        pessimistic = results[-1].intrinsic_value if results else 0

        upside_risk = optimistic - expected_value if optimistic > expected_value else 0
        downside_risk = expected_value - pessimistic if pessimistic < expected_value else 0

        # 期望上行空间
        expected_upside = (
            (expected_value - current_price) / current_price * 100
            if current_price > 0 else 0.0
        )

        summary = {
            "stock_code": stock_code,
            "current_price": round(current_price, 2),
            "expected_value": round(expected_value, 2),
            "expected_upside_pct": round(expected_upside, 2),
            "upside_risk": round(upside_risk, 2),
            "downside_risk": round(downside_risk, 2),
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "intrinsic_value": r.intrinsic_value,
                    "upside_pct": r.upside_pct,
                    "probability": round(prob, 3),
                }
                for r, prob in zip(results, scenario_probabilities)
            ],
        }

        logger.info(
            f"DCF期望估值: {stock_code}, "
            f"期望价值={expected_value:.2f}, "
            f"上行={summary['expected_upside_pct']:.1f}%"
        )
        return summary

    def value_dataframe(
        self,
        df: pd.DataFrame,
        stock_col: str = "stock_code",
        revenue_col: str = "revenue",
        margin_col: str = "gross_margin",
        price_col: str = "close_price",
        shares_col: str = "total_equity",
    ) -> pd.DataFrame:
        """
        对 DataFrame 中的多只股票进行批量估值。

        Parameters
        ----------
        df : pd.DataFrame
            包含财务数据的股票表
        stock_col : str
            股票代码列
        revenue_col : str
            营收列
        margin_col : str
            利润率列
        price_col : str
            当前股价列
        shares_col : str
            总股本列

        Returns
        -------
        pd.DataFrame
            估值结果汇总表
        """
        all_results = []
        for _, row in df.iterrows():
            try:
                result = self.value_expected(
                    stock_code=row.get(stock_col, ""),
                    base_revenue=row.get(revenue_col, 0),
                    base_margin=row.get(margin_col, 0),
                    current_price=row.get(price_col, 0),
                    total_shares=max(row.get(shares_col, 1), 0.01),
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"估值失败 [{row.get(stock_col, '?')}]: {e}")
                continue

        result_df = pd.DataFrame(all_results)
        logger.info(f"批量估值完成: {len(result_df)} 只股票")
        return result_df


# ============================================================================
# 便捷函数
# ============================================================================

def quick_dcf(
    revenue: float,
    margin: float,
    wacc: float = 0.10,
    growth: float = 0.08,
    shares: float = 1.0,
    years: int = 5,
) -> float:
    """
    便捷函数：快速 DCF 估值。

    Parameters
    ----------
    revenue : float
        基准营收
    margin : float
        净利润率
    wacc : float
        折现率
    growth : float
        增长率
    shares : float
        总股本
    years : int
        预测期

    Returns
    -------
    float
        每股内在价值
    """
    scenario = ScenarioParams("quick", growth, 0.0, wacc, 0.025)
    valuator = DCFValuator(forecast_years=years, scenarios=[scenario])
    result = valuator.value_single_scenario(
        "quick", revenue, margin, 0, shares, scenario, 0
    )
    return result.intrinsic_value
