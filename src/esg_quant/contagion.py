"""
行业风险传导分析模块
====================
基于投入产出关系和行业关联矩阵，模拟 ESG 风险在产业链中的传导效应。

模型假设：
  1. 风险从上游向下游传导（供给侧冲击）
  2. 传导强度随产业链距离指数衰减
  3. 行业自身的ESG韧性可缓冲传导冲击
  4. 最终输出每个行业的总风险暴露（自身风险 + 传导风险）

应用场景：
  - 上游行业ESG违规 → 下游行业成本上升 → 估值下修
  - 碳税政策 → 高碳行业 → 全产业链价格传导
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 传导分析器
# ============================================================================

class ContagionAnalyzer:
    """
    行业 ESG 风险传导分析器。

    基于有向加权图（行业 → 行业的传导系数），
    使用 Floyd-Warshall 类算法计算所有行业间的最短传导路径和衰减后强度。

    Attributes
    ----------
    linkages : list of dict
        行业关联边列表 [{source, target, coefficient}, ...]
    matrix : pd.DataFrame
        传导系数矩阵（行业 × 行业）
    industries : list of str
        已知行业列表
    max_distance : int
        最大传导距离
    decay_factor : float
        每步距离的衰减因子
    """

    def __init__(
        self,
        max_distance: int = 5,
        decay_factor: float = 0.5,
    ) -> None:
        """
        初始化传导分析器。

        Parameters
        ----------
        max_distance : int
            最大传导链长度（超过此距离的传导忽略不计）
        decay_factor : float
            每步传导衰减因子（0~1），0.5 表示每步衰减50%
        """
        self.max_distance = max_distance
        self.decay_factor = decay_factor
        self.linkages: List[Dict[str, Any]] = []
        self.matrix: pd.DataFrame = pd.DataFrame()
        self.industries: List[str] = []
        self._risk_cache: Dict[str, pd.DataFrame] = {}
        logger.info(
            f"ContagionAnalyzer 初始化: max_distance={max_distance}, "
            f"decay_factor={decay_factor}"
        )

    def load_linkages(
        self, config_data: List[Dict[str, Any]]
    ) -> None:
        """
        从配置加载行业关联数据。

        Parameters
        ----------
        config_data : list of dict
            行业关联配置
            每项格式: {"source": str, "target": str, "coefficient": float}
        """
        self.linkages = config_data

        # 收集所有行业
        sources = {item["source"] for item in config_data}
        targets = {item["target"] for item in config_data}
        self.industries = sorted(sources | targets)

        # 构建直接传导矩阵
        n = len(self.industries)
        idx_map = {ind: i for i, ind in enumerate(self.industries)}

        matrix = np.zeros((n, n))
        for item in config_data:
            i = idx_map[item["source"]]
            j = idx_map[item["target"]]
            matrix[i, j] = item["coefficient"]

        self.matrix = pd.DataFrame(
            matrix, index=self.industries, columns=self.industries
        )
        logger.info(
            f"关联矩阵加载完成: {len(self.industries)} 个行业, "
            f"{len(config_data)} 条关联边"
        )

    def compute_all_pairs_impact(self) -> pd.DataFrame:
        """
        计算任意两个行业间的风险传导总影响。

        使用多步传导模型：
          total_impact(i→j) = max over paths {
              Π coefficient_k × decay_factor^(k-1)
          }
        即所有可能传导路径中影响力最大的那条。

        Returns
        -------
        pd.DataFrame
            总影响矩阵（行业 × 行业），C_ij 表示行业 i 对行业 j 的总传导影响
        """
        if self.matrix.empty:
            logger.error("关联矩阵为空，请先调用 load_linkages()")
            return pd.DataFrame()

        n = len(self.industries)
        # 使用类似 Floyd-Warshall 的 DP
        # impact[k][i][j] = i到j经过至多k步的最大影响
        impact = self.matrix.values.copy()

        # 多步传导
        current = impact.copy()
        for step in range(2, self.max_distance + 1):
            # current 是精确 step-1 步的影响
            decay = self.decay_factor ** (step - 1)
            next_step = np.zeros((n, n))
            for k in range(n):
                # i → k → j
                col_k = current[:, k:k+1]  # (n, 1)
                row_k = self.matrix.values[k:k+1, :]  # (1, n)
                contrib = col_k @ row_k * decay
                next_step = np.maximum(next_step, contrib)
            current = next_step
            impact = np.maximum(impact, current)

        # 对角线为0（自身不传导）
        np.fill_diagonal(impact, 0.0)

        result = pd.DataFrame(
            impact, index=self.industries, columns=self.industries
        ).round(4)

        logger.info(f"全对传导影响计算完成: {n}×{n} 矩阵")
        return result

    def compute_exposure(
        self,
        risk_scores: Dict[str, float],
        esg_resilience: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        计算各行业的总 ESG 风险暴露。

        总暴露 = 自身风险 + Σ(上游风险 × 传导系数 × 衰减) × (1 - 韧性)

        Parameters
        ----------
        risk_scores : dict
            各行业的自身ESG风险评分 {行业: 风险分 (0-100)}
        esg_resilience : dict, optional
            各行业的ESG韧性系数 {行业: 韧性 (0-1)}，越高越能缓冲冲击

        Returns
        -------
        pd.DataFrame
            风险暴露明细表
        """
        if self.matrix.empty:
            logger.error("关联矩阵为空")
            return pd.DataFrame()

        all_impact = self.compute_all_pairs_impact()

        results = []
        for industry in self.industries:
            own_risk = risk_scores.get(industry, 0.0)

            # 来自上游的传导风险
            upstream_col = all_impact[industry]  # 所有源 → 当前行业
            contagion_risk = 0.0
            contagion_sources: List[Tuple[str, float]] = []

            for source in self.industries:
                if source == industry:
                    continue
                source_risk = risk_scores.get(source, 0.0)
                impact_coeff = upstream_col.get(source, 0.0)
                contribution = source_risk * impact_coeff
                if contribution > 0.001:
                    contagion_risk += contribution
                    contagion_sources.append((source, round(contribution, 2)))

            # 韧性缓冲
            resilience = esg_resilience.get(industry, 0.3) if esg_resilience else 0.3
            buffered_contagion = contagion_risk * (1 - resilience)

            total_exposure = own_risk + buffered_contagion

            results.append({
                "industry": industry,
                "own_risk": round(own_risk, 2),
                "contagion_risk_raw": round(contagion_risk, 2),
                "contagion_risk_buffered": round(buffered_contagion, 2),
                "resilience": round(resilience, 2),
                "total_exposure": round(total_exposure, 2),
                "risk_amplification": round(
                    total_exposure / max(own_risk, 0.01), 2
                ),
                "top_contagion_sources": contagion_sources[:3],
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(
            "total_exposure", ascending=False
        ).reset_index(drop=True)

        logger.info(
            f"风险暴露计算完成: 最大暴露={result_df['total_exposure'].max():.2f}, "
            f"平均放大倍数={result_df['risk_amplification'].mean():.2f}"
        )
        return result_df

    def find_contagion_paths(
        self, source: str, target: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        查找从 source 到 target 的最强传导路径。

        使用 BFS + 优先级队列，找到影响力最大的 top_k 条路径。

        Parameters
        ----------
        source : str
            上游行业
        target : str
            下游行业
        top_k : int
            返回路径数

        Returns
        -------
        list of dict
            路径列表 [{path: [...], strength: float, steps: int}, ...]
        """
        import heapq

        if source not in self.industries or target not in self.industries:
            logger.warning(f"行业 {source} 或 {target} 不在已知列表中")
            return []

        paths: List[Dict[str, Any]] = []
        # 小顶堆（存负值实现大顶堆）
        heap: List[Tuple[float, List[str]]] = [(-1.0, [source])]

        while heap and len(paths) < top_k * 3:
            neg_strength, path = heapq.heappop(heap)
            strength = -neg_strength
            current = path[-1]

            if len(path) > self.max_distance + 1:
                continue

            if current == target and len(path) > 1:
                paths.append({
                    "path": path,
                    "strength": round(strength, 4),
                    "steps": len(path) - 1,
                })
                continue

            # 扩展邻居
            for neighbor in self.industries:
                if neighbor in path:
                    continue  # 避免环
                coeff = self.matrix.loc[current, neighbor]
                if coeff > 0:
                    new_strength = strength * coeff * self.decay_factor
                    heapq.heappush(heap, (-new_strength, path + [neighbor]))

        # 去重并取 top_k
        seen = set()
        unique_paths = []
        for p in paths:
            key = tuple(p["path"])
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        result = sorted(unique_paths, key=lambda x: x["strength"], reverse=True)[:top_k]
        return result

    def sensitivity_analysis(
        self,
        shock_industry: str,
        shock_magnitude: float = 10.0,
    ) -> pd.DataFrame:
        """
        敏感性分析：对某一行业施加冲击，观察全产业链影响。

        Parameters
        ----------
        shock_industry : str
            施加冲击的行业
        shock_magnitude : float
            冲击强度（风险分增量）

        Returns
        -------
        pd.DataFrame
            各行业受影响程度排名
        """
        all_impact = self.compute_all_pairs_impact()

        # 从冲击源到所有行业的传导影响
        impact_series = all_impact.loc[shock_industry]
        affected = impact_series[impact_series > 0.001].sort_values(ascending=False)

        results = []
        for industry, coeff in affected.items():
            if industry == shock_industry:
                continue
            direct = (
                self.matrix.loc[shock_industry, industry]
                if industry in self.matrix.columns
                else 0.0
            )
            results.append({
                "target_industry": industry,
                "shock_impact": round(shock_magnitude * coeff, 2),
                "transmission_coeff": round(coeff, 4),
                "is_direct": direct > 0,
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("shock_impact", ascending=False).reset_index(
            drop=True
        )

        logger.info(
            f"敏感性分析: 冲击源={shock_industry}, "
            f"影响 {len(result_df)} 个行业, "
            f"最大冲击={result_df['shock_impact'].max():.2f}"
        )
        return result_df


# ============================================================================
# 风险矩阵可视化数据生成
# ============================================================================

def generate_contagion_heatmap_data(
    analyzer: ContagionAnalyzer,
) -> Dict[str, Any]:
    """
    生成行业传导矩阵的热图数据（供 visualizer 使用）。

    Parameters
    ----------
    analyzer : ContagionAnalyzer
        已加载数据的传导分析器

    Returns
    -------
    dict
        {"industries": [...], "matrix": [[...], ...], "annotations": [...]}
    """
    if analyzer.matrix.empty:
        return {"industries": [], "matrix": [], "annotations": []}

    all_impact = analyzer.compute_all_pairs_impact()
    industries = analyzer.industries

    return {
        "industries": industries,
        "matrix": all_impact.values.tolist(),
        "annotations": [
            [f"{v:.2f}" if v > 0 else "" for v in row]
            for row in all_impact.values
        ],
    }


def compute_industry_risk_network(
    risk_scores: Dict[str, float],
    linkages: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    便捷函数：一键计算行业风险网络。

    Parameters
    ----------
    risk_scores : dict
        行业ESG风险评分
    linkages : list of dict
        行业关联配置

    Returns
    -------
    pd.DataFrame
        风险暴露表
    """
    analyzer = ContagionAnalyzer()
    analyzer.load_linkages(linkages)
    return analyzer.compute_exposure(risk_scores)
