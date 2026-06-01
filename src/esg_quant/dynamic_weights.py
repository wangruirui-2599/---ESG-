"""
动态行业权重引擎
================
在行业基础权重表之上，叠加政策调控信号和舆情事件信号，
实现 ESG 三维度 (E/S/G) 权重的动态调整。

核心逻辑：
  1. 加载基础权重表（config/industry_weights.yaml）
  2. 接收政策强度系数（如"双碳政策"使E权重+10%）
  3. 接收舆情热度系数（如"员工权益事件"使S权重+5%）
  4. 输出调整后的行业动态权重，保证 E+S+G=1.0

应用场景：
  - 政策密集期：环境(E)权重上调，反映合规风险
  - 社会事件期：社会(S)权重上调，反映声誉风险
  - 治理丑闻期：治理(G)权重上调，反映代理风险
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# 配置模型
# ============================================================================

class PolicySignal(BaseModel):
    """
    政策调控信号。

    定义某一政策事件对各ESG维度权重的调整方向与幅度。
    """
    name: str = Field(..., description="政策/事件名称")
    dimension: str = Field(..., description="影响的ESG维度 (E/S/G)")
    adjustment: float = Field(default=0.0, description="权重调整幅度 (-0.2 ~ +0.2)")
    affected_industries: List[str] = Field(
        default_factory=list, description="受影响的行业列表（空=全部）"
    )
    decay_rate: float = Field(default=0.3, description="月度衰减率 (0~1)")

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: str) -> str:
        if v not in ("E", "S", "G"):
            raise ValueError(f"ESG维度必须为 E/S/G，收到: {v}")
        return v

    @field_validator("adjustment")
    @classmethod
    def validate_adjustment(cls, v: float) -> float:
        if not -0.5 <= v <= 0.5:
            raise ValueError(f"调整幅度应在 [-0.5, 0.5] 范围内，收到: {v}")
        return v


class DynamicWeightResult(BaseModel):
    """动态权重输出"""
    industry: str
    base_E: float
    base_S: float
    base_G: float
    adjusted_E: float
    adjusted_S: float
    adjusted_G: float
    E_shift: float
    S_shift: float
    G_shift: float


# ============================================================================
# 动态权重引擎
# ============================================================================

class DynamicWeightEngine:
    """
    动态 ESG 权重引擎。

    在基础行业权重上叠加政策信号和舆情信号，
    通过指数衰减模型对历史信号进行时间加权，
    输出每个行业当前的动态 ESG 三维度权重。

    Attributes
    ----------
    base_weights : pd.DataFrame
        基础权重表（行业 × E/S/G 权重）
    policy_signals : list of PolicySignal
        活跃的政策调控信号
    sentiment_signals : list of PolicySignal
        舆情事件信号
    """

    def __init__(self) -> None:
        """初始化动态权重引擎。"""
        self.base_weights: pd.DataFrame = pd.DataFrame()
        self.policy_signals: List[PolicySignal] = []
        self.sentiment_signals: List[PolicySignal] = []
        self._signal_history: List[Dict[str, Any]] = []
        logger.info("DynamicWeightEngine 初始化完成")

    def load_base_weights(
        self, config_data: List[Dict[str, Any]]
    ) -> None:
        """
        从配置加载基础权重表。

        Parameters
        ----------
        config_data : list of dict
            行业权重配置列表
            每项格式: {"industry": str, "E_weight": float, "S_weight": float, "G_weight": float}
        """
        records = []
        for item in config_data:
            # 归一化确保和为1
            total = item["E_weight"] + item["S_weight"] + item["G_weight"]
            if abs(total - 1.0) > 1e-6:
                logger.debug(f"行业 {item['industry']} 权重之和={total:.3f}, 自动归一化")
                item["E_weight"] /= total
                item["S_weight"] /= total
                item["G_weight"] /= total

            records.append({
                "industry": item["industry"],
                "E_weight": round(item["E_weight"], 4),
                "S_weight": round(item["S_weight"], 4),
                "G_weight": round(item["G_weight"], 4),
            })

        self.base_weights = pd.DataFrame(records)
        logger.info(f"基础权重加载完成: {len(self.base_weights)} 个行业")

    def register_policy_signal(self, signal: PolicySignal) -> None:
        """
        注册政策调控信号。

        Parameters
        ----------
        signal : PolicySignal
            政策信号对象
        """
        self.policy_signals.append(signal)
        self._signal_history.append({
            "type": "policy",
            "name": signal.name,
            "dimension": signal.dimension,
            "adjustment": signal.adjustment,
        })
        logger.info(
            f"政策信号已注册: {signal.name} → "
            f"{signal.dimension}维度 {'+' if signal.adjustment >= 0 else ''}{signal.adjustment:.1%}"
        )

    def register_sentiment_signal(self, signal: PolicySignal) -> None:
        """
        注册舆情事件信号。

        Parameters
        ----------
        signal : PolicySignal
            舆情信号对象
        """
        self.sentiment_signals.append(signal)
        self._signal_history.append({
            "type": "sentiment",
            "name": signal.name,
            "dimension": signal.dimension,
            "adjustment": signal.adjustment,
        })
        logger.info(
            f"舆情信号已注册: {signal.name} → "
            f"{signal.dimension}维度 {'+' if signal.adjustment >= 0 else ''}{signal.adjustment:.1%}"
        )

    def clear_signals(self) -> None:
        """清除所有活跃信号。"""
        self.policy_signals.clear()
        self.sentiment_signals.clear()
        logger.info("所有信号已清除")

    def compute_dynamic_weights(
        self,
        months_elapsed: float = 0.0,
    ) -> pd.DataFrame:
        """
        计算当前所有行业的动态权重。

        算法：
        1. 从基础权重出发
        2. 叠加每个政策信号的调整
        3. 叠加每个舆情信号的调整
        4. 每次叠加以 decay_rate 进行时间衰减
        5. 最终归一化确保 E+S+G=1.0

        Parameters
        ----------
        months_elapsed : float
            信号自发布以来的月数（用于衰减计算）

        Returns
        -------
        pd.DataFrame
            动态权重表，包含 基础值/调整值/偏移量
        """
        if self.base_weights.empty:
            logger.error("基础权重表为空，请先调用 load_base_weights()")
            return pd.DataFrame()

        result = self.base_weights.copy()
        result["E_adjusted"] = result["E_weight"].copy()
        result["S_adjusted"] = result["S_weight"].copy()
        result["G_adjusted"] = result["G_weight"].copy()

        # 合并所有信号
        all_signals = self.policy_signals + self.sentiment_signals

        for signal in all_signals:
            decay_factor = np.exp(-signal.decay_rate * months_elapsed)
            effective_adjustment = signal.adjustment * decay_factor

            # 确定影响范围
            if signal.affected_industries:
                mask = result["industry"].isin(signal.affected_industries)
            else:
                mask = pd.Series(True, index=result.index)

            # 应用调整
            dim_col = f"{signal.dimension}_adjusted"
            result.loc[mask, dim_col] += effective_adjustment

            # 从其他两个维度扣除以确保和不变（按比例分配）
            other_dims = [d for d in ["E", "S", "G"] if d != signal.dimension]
            for other in other_dims:
                other_col = f"{other}_adjusted"
                # 按该维度原始权重比例分配扣除额
                other_ratio = result.loc[mask, f"{other}_weight"] / (
                    result.loc[mask, [f"{d}_weight" for d in other_dims]].sum(axis=1)
                ).replace(0, 1.0)
                result.loc[mask, other_col] -= effective_adjustment * other_ratio.values

        # 确保非负并归一化
        for dim in ["E", "S", "G"]:
            result[f"{dim}_adjusted"] = result[f"{dim}_adjusted"].clip(0.05, 0.90)

        # 归一化到和为1
        row_sums = result[["E_adjusted", "S_adjusted", "G_adjusted"]].sum(axis=1)
        for dim in ["E", "S", "G"]:
            result[f"{dim}_adjusted"] = (
                result[f"{dim}_adjusted"] / row_sums.replace(0, 1.0)
            ).round(4)

        # 计算偏移量
        result["E_shift"] = (result["E_adjusted"] - result["E_weight"]).round(4)
        result["S_shift"] = (result["S_adjusted"] - result["S_weight"]).round(4)
        result["G_shift"] = (result["G_adjusted"] - result["G_weight"]).round(4)

        logger.info(
            f"动态权重计算完成: {len(result)} 个行业, "
            f"E平均偏移={result['E_shift'].mean():.4f}, "
            f"S平均偏移={result['S_shift'].mean():.4f}, "
            f"G平均偏移={result['G_shift'].mean():.4f}"
        )

        return result

    def get_weights_for_industry(
        self, industry: str, months_elapsed: float = 0.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        获取指定行业的动态权重。

        Parameters
        ----------
        industry : str
            行业名称
        months_elapsed : float
            信号距今月数

        Returns
        -------
        tuple of (E, S, G) or None
            三元组权重，行业不存在时返回 None
        """
        all_weights = self.compute_dynamic_weights(months_elapsed)
        row = all_weights[all_weights["industry"] == industry]
        if row.empty:
            logger.warning(f"行业 '{industry}' 不在权重表中")
            return None
        return (
            row["E_adjusted"].values[0],
            row["S_adjusted"].values[0],
            row["G_adjusted"].values[0],
        )

    def get_weight_shift_matrix(self) -> pd.DataFrame:
        """
        获取权重偏移矩阵（用于风险传导分析的输入）。

        Returns
        -------
        pd.DataFrame
            行业 × 维度偏移矩阵
        """
        weights = self.compute_dynamic_weights()
        return weights[["industry", "E_shift", "S_shift", "G_shift"]].copy()

    def export_to_dict(self) -> List[Dict[str, Any]]:
        """
        导出动态权重为字典列表。

        Returns
        -------
        list of dict
        """
        weights = self.compute_dynamic_weights()
        return weights.to_dict(orient="records")

    def build_industry_weight_vector(
        self, df: pd.DataFrame, industry_col: str = "industry"
    ) -> pd.DataFrame:
        """
        为 DataFrame 中的每行附加对应的动态权重。

        Parameters
        ----------
        df : pd.DataFrame
            待附加的数据表（必须包含行业列）
        industry_col : str
            行业列名

        Returns
        -------
        pd.DataFrame
            添加了 dyn_E_weight, dyn_S_weight, dyn_G_weight 列的数据表
        """
        weights = self.compute_dynamic_weights()
        weight_map = weights.set_index("industry")[
            ["E_adjusted", "S_adjusted", "G_adjusted"]
        ].to_dict(orient="index")

        df = df.copy()
        df["dyn_E_weight"] = df[industry_col].map(
            lambda x: weight_map.get(x, {}).get("E_adjusted", 0.33)
        )
        df["dyn_S_weight"] = df[industry_col].map(
            lambda x: weight_map.get(x, {}).get("S_adjusted", 0.33)
        )
        df["dyn_G_weight"] = df[industry_col].map(
            lambda x: weight_map.get(x, {}).get("G_adjusted", 0.34)
        )

        return df


# ============================================================================
# 便捷函数
# ============================================================================

def create_default_signals() -> List[PolicySignal]:
    """
    创建一组示例政策信号（用于演示和测试）。

    Returns
    -------
    list of PolicySignal
    """
    return [
        PolicySignal(
            name="双碳政策强化",
            dimension="E",
            adjustment=0.10,
            affected_industries=["石油石化", "煤炭", "钢铁", "公用事业", "基础化工"],
            decay_rate=0.10,
        ),
        PolicySignal(
            name="ESG信息披露新规",
            dimension="G",
            adjustment=0.08,
            affected_industries=[],  # 全行业
            decay_rate=0.08,
        ),
        PolicySignal(
            name="劳动权益保障专项行动",
            dimension="S",
            adjustment=0.06,
            affected_industries=["商贸零售", "社会服务", "纺织服装", "食品饮料"],
            decay_rate=0.15,
        ),
    ]
