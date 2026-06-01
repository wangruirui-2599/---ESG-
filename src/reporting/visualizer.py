"""
可视化图表生成模块
==================
基于 Matplotlib、Seaborn 和 Plotly 生成专业的金融分析图表。

图表清单：
  1. ESG三维度雷达图
  2. 行业权重热力图
  3. DCF情景估值瀑布图
  4. 四因子贡献饼图/柱状图
  5. 异常概率分布直方图
  6. 组合净值曲线
  7. 风险传导网络图（Plotly）
  8. ESG趋势动量散点图
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

import matplotlib
matplotlib.use("Agg")  # 非交互式后端

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


# ============================================================================
# 全局样式
# ============================================================================

COLOR_PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "green": "#2ECC71",
    "red": "#E74C3C",
    "blue": "#3498DB",
    "orange": "#E67E22",
    "purple": "#9B59B6",
    "gray": "#95A5A6",
    "dark": "#2C3E50",
}

ESG_COLORS = {
    "E": "#2ECC71",  # 绿色=环境
    "S": "#3498DB",  # 蓝色=社会
    "G": "#9B59B6",  # 紫色=治理
}

DEFAULT_FIGSIZE = (12, 7)
DEFAULT_DPI = 150


def _ensure_output_dir(output_dir: str) -> Path:
    """确保输出目录存在。"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# 1. ESG 雷达图
# ============================================================================

def plot_esg_radar(
    e_score: float,
    s_score: float,
    g_score: float,
    esg_total: float = 0.0,
    industry: str = "",
    stock_code: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 ESG 三维度雷达图。

    Parameters
    ----------
    e_score, s_score, g_score : float
        各维度评分 (0-100)
    esg_total : float
        ESG 总分
    industry : str
        行业名称
    stock_code : str
        股票代码
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    categories = ["环境 (E)", "社会 (S)", "治理 (G)"]
    values = [e_score, s_score, g_score]
    values += values[:1]  # 闭合

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 绘制背景网格
    ax.set_rlabel_position(30)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    # 填充区域
    ax.fill(angles, values, color=COLOR_PALETTE["primary"], alpha=0.25)
    ax.plot(angles, values, color=COLOR_PALETTE["primary"], linewidth=2, marker="o", markersize=8)

    # 标注数值
    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.annotate(
            f"{value:.0f}",
            xy=(angle, value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    title = f"ESG 评分雷达图"
    if stock_code:
        title += f" - {stock_code}"
    if industry:
        title += f" ({industry})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"雷达图已保存: {output_path}")

    return fig


# ============================================================================
# 2. 行业权重热力图
# ============================================================================

def plot_industry_weight_heatmap(
    weight_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制行业 ESG 权重热力图。

    Parameters
    ----------
    weight_df : pd.DataFrame
        权重表，包含 industry, E_weight, S_weight, G_weight 列
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, max(8, len(weight_df) * 0.35)))

    heatmap_data = weight_df.set_index("industry")[
        ["E_weight", "S_weight", "G_weight"]
    ]

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=0.6,
        linewidths=0.5,
        cbar_kws={"label": "权重"},
        ax=ax,
    )

    ax.set_title("行业 ESG 维度权重分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("ESG 维度", fontsize=11)
    ax.set_ylabel("行业", fontsize=11)

    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"热力图已保存: {output_path}")

    return fig


# ============================================================================
# 3. DCF 情景估值瀑布图
# ============================================================================

def plot_dcf_scenario_waterfall(
    scenario_results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 DCF 多情景估值对比图。

    Parameters
    ----------
    scenario_results : dict
        DCF估值结果（来自 ScenarioValuator.value_expected()）
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：情景估值柱状图
    scenarios = scenario_results.get("scenarios", [])
    names = [s["name"] for s in scenarios]
    ivalues = [s["intrinsic_value"] for s in scenarios]
    probs = [s["probability"] for s in scenarios]

    colors = [COLOR_PALETTE["green"], COLOR_PALETTE["blue"], COLOR_PALETTE["red"]]
    axes[0].bar(names, ivalues, color=colors[:len(names)], alpha=0.8, edgecolor="white")
    axes[0].axhline(
        y=scenario_results.get("current_price", 0),
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"当前价格 {scenario_results.get('current_price', 0):.2f}",
    )
    axes[0].axhline(
        y=scenario_results.get("expected_value", 0),
        color=COLOR_PALETTE["primary"],
        linestyle="-",
        linewidth=2,
        label=f"期望估值 {scenario_results.get('expected_value', 0):.2f}",
    )

    # 标注概率
    for i, (v, p) in enumerate(zip(ivalues, probs)):
        axes[0].annotate(
            f"P={p:.0%}", (i, v), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=10,
        )

    axes[0].set_title("DCF 多情景估值对比", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("每股内在价值（元）", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)

    # 右图：概率饼图
    wedges, texts, autotexts = axes[1].pie(
        probs, labels=names, autopct="%1.1f%%",
        colors=colors[:len(names)],
        startangle=90, explode=(0.05, 0, 0.05),
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[1].set_title("情景概率分布", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"瀑布图已保存: {output_path}")

    return fig


# ============================================================================
# 4. 四因子贡献饼图
# ============================================================================

def plot_factor_contribution(
    fusion_result: Dict[str, Any],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制四因子贡献可视化。

    Parameters
    ----------
    fusion_result : dict
        四因子融合结果
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：贡献柱状图
    contribs = {
        "DCF估值": fusion_result.get("dcf_contrib", 0),
        "相对估值": fusion_result.get("relative_contrib", 0),
        "ESG因子": fusion_result.get("esg_contrib", 0),
        "市场情绪": fusion_result.get("sentiment_contrib", 0),
    }
    names = list(contribs.keys())
    values = list(contribs.values())
    colors = [COLOR_PALETTE["primary"], COLOR_PALETTE["accent"],
              COLOR_PALETTE["green"], COLOR_PALETTE["purple"]]

    bars = axes[0].bar(names, values, color=colors, alpha=0.85, edgecolor="white")
    axes[0].axhline(
        y=fusion_result.get("final_value", 0),
        color="black", linestyle="--", linewidth=2,
        label=f"综合估值: {fusion_result.get('final_value', 0):.2f}",
    )

    # 标注数值
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
            f"{val:.2f}", ha="center", fontsize=10,
        )

    axes[0].set_title("四因子贡献分解", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("估值贡献（元）", fontsize=11)
    axes[0].legend(fontsize=9)

    # 右图：占比饼图
    pcts = [
        fusion_result.get("dcf_contrib_pct", 25),
        fusion_result.get("relative_contrib_pct", 25),
        fusion_result.get("esg_contrib_pct", 25),
        fusion_result.get("sentiment_contrib_pct", 25),
    ]
    axes[1].pie(
        pcts, labels=names, autopct="%1.1f%%",
        colors=colors, startangle=90,
    )
    axes[1].set_title("因子贡献占比", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"因子贡献图已保存: {output_path}")

    return fig


# ============================================================================
# 5. 异常概率分布直方图
# ============================================================================

def plot_anomaly_distribution(
    anomaly_probs: pd.Series,
    risk_levels: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制异常概率分布直方图。

    Parameters
    ----------
    anomaly_probs : pd.Series
        异常概率序列
    risk_levels : pd.Series, optional
        风险等级标签
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # 直方图
    n, bins, patches = ax.hist(
        anomaly_probs, bins=30, alpha=0.7,
        color=COLOR_PALETTE["secondary"], edgecolor="white",
    )

    # 风险区域着色
    ax.axvspan(0, 0.2, alpha=0.1, color="green", label="低风险 (<0.2)")
    ax.axvspan(0.2, 0.4, alpha=0.1, color="yellow", label="较低风险 (0.2-0.4)")
    ax.axvspan(0.4, 0.6, alpha=0.1, color="orange", label="中等风险 (0.4-0.6)")
    ax.axvspan(0.6, 0.8, alpha=0.1, color="orangered", label="较高风险 (0.6-0.8)")
    ax.axvspan(0.8, 1.0, alpha=0.1, color="red", label="高风险 (>0.8)")

    ax.axvline(
        x=anomaly_probs.mean(), color="black", linestyle="--", linewidth=2,
        label=f"均值: {anomaly_probs.mean():.3f}",
    )
    ax.axvline(
        x=anomaly_probs.median(), color="gray", linestyle=":", linewidth=2,
        label=f"中位数: {anomaly_probs.median():.3f}",
    )

    ax.set_xlabel("异常概率", fontsize=11)
    ax.set_ylabel("频数", fontsize=11)
    ax.set_title("财务异常概率分布", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"异常分布图已保存: {output_path}")

    return fig


# ============================================================================
# 6. 组合净值曲线
# ============================================================================

def plot_portfolio_nav(
    nav_data: pd.DataFrame,
    benchmark_nav: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制组合净值曲线与回撤。

    Parameters
    ----------
    nav_data : pd.DataFrame
        组合净值数据（含 day, nav 列）
    benchmark_nav : pd.DataFrame, optional
        基准净值数据
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})

    # 上图：净值曲线
    axes[0].plot(
        nav_data["day"], nav_data["nav"],
        color=COLOR_PALETTE["primary"], linewidth=2, label="策略组合",
    )
    if benchmark_nav is not None:
        axes[0].plot(
            benchmark_nav["day"], benchmark_nav["nav"],
            color=COLOR_PALETTE["gray"], linewidth=1.5, linestyle="--",
            label="基准指数",
        )

    axes[0].axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    axes[0].fill_between(
        nav_data["day"], 1, nav_data["nav"],
        where=(nav_data["nav"] >= 1),
        color=COLOR_PALETTE["green"], alpha=0.15,
    )
    axes[0].fill_between(
        nav_data["day"], 1, nav_data["nav"],
        where=(nav_data["nav"] < 1),
        color=COLOR_PALETTE["red"], alpha=0.15,
    )
    axes[0].set_title("组合净值曲线", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("净值", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=10)

    # 下图：回撤曲线
    nav = nav_data["nav"].values
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak * 100

    axes[1].fill_between(
        nav_data["day"], 0, drawdown,
        color=COLOR_PALETTE["red"], alpha=0.4,
    )
    axes[1].plot(
        nav_data["day"], drawdown,
        color=COLOR_PALETTE["red"], linewidth=1.5,
    )
    axes[1].set_ylabel("回撤 (%)", fontsize=11)
    axes[1].set_xlabel("交易日", fontsize=11)
    axes[1].axhline(y=0, color="black", linestyle=":", linewidth=0.5)

    # 最大回撤标注
    max_dd_idx = np.argmin(drawdown)
    axes[1].annotate(
        f"最大回撤: {drawdown[max_dd_idx]:.1f}%",
        xy=(max_dd_idx, drawdown[max_dd_idx]),
        xytext=(max_dd_idx + 10, drawdown[max_dd_idx] - 5),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10, fontweight="bold",
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"净值曲线已保存: {output_path}")

    return fig


# ============================================================================
# 7. ESG 趋势散点图
# ============================================================================

def plot_esg_trend_scatter(
    df: pd.DataFrame,
    x_col: str = "ESG_total_momentum",
    y_col: str = "trend_score",
    label_col: str = "trend_label",
    stock_col: str = "stock_code",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 ESG 趋势-动量散点图。

    Parameters
    ----------
    df : pd.DataFrame
        趋势分析结果
    x_col : str
        X轴（动量）
    y_col : str
        Y轴（趋势分）
    label_col : str
        分类标签列
    stock_col : str
        股票代码列
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    label_colors = {
        "显著改善": COLOR_PALETTE["green"],
        "改善": "#27AE60",
        "稳定": COLOR_PALETTE["gray"],
        "恶化": COLOR_PALETTE["orange"],
        "显著恶化": COLOR_PALETTE["red"],
        "波动": COLOR_PALETTE["purple"],
    }

    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        color = label_colors.get(label, COLOR_PALETTE["blue"])
        ax.scatter(
            subset[x_col], subset[y_col],
            c=color, label=label, alpha=0.6, s=60, edgecolors="white",
        )

    # 标注极端值
    if len(df) > 0:
        top_n = min(5, len(df))
        top_improvers = df.nlargest(top_n, y_col)
        top_decliners = df.nsmallest(top_n, y_col)

        for _, row in pd.concat([top_improvers, top_decliners]).iterrows():
            ax.annotate(
                str(row.get(stock_col, "")),
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5), fontsize=8, alpha=0.8,
            )

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("ESG 动量分数", fontsize=11)
    ax.set_ylabel("ESG 趋势分数", fontsize=11)
    ax.set_title("ESG 趋势分析散点图", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"趋势散点图已保存: {output_path}")

    return fig


# ============================================================================
# 8. 综合仪表板
# ============================================================================

def plot_dashboard(
    esg_radar_data: Dict[str, float],
    scenario_data: Dict[str, Any],
    anomaly_data: pd.Series,
    output_dir: str = "output/figures",
) -> Dict[str, str]:
    """
    生成综合可视化仪表板（所有图表）。

    Parameters
    ----------
    esg_radar_data : dict
        ESG评分数据
    scenario_data : dict
        DCF估值结果
    anomaly_data : pd.Series
        异常概率序列
    output_dir : str
        图表输出目录

    Returns
    -------
    dict
        图表文件路径映射
    """
    out = _ensure_output_dir(output_dir)
    paths = {}

    # 1. ESG雷达图
    paths["radar"] = str(out / "esg_radar.png")
    plot_esg_radar(
        esg_radar_data.get("E_score", 0),
        esg_radar_data.get("S_score", 0),
        esg_radar_data.get("G_score", 0),
        esg_radar_data.get("ESG_total", 0),
        output_path=paths["radar"],
    )
    plt.close("all")

    # 2. DCF情景图
    paths["scenario"] = str(out / "dcf_scenario.png")
    plot_dcf_scenario_waterfall(scenario_data, output_path=paths["scenario"])
    plt.close("all")

    # 3. 异常分布图
    paths["anomaly"] = str(out / "anomaly_dist.png")
    plot_anomaly_distribution(anomaly_data, output_path=paths["anomaly"])
    plt.close("all")

    logger.info(f"仪表板生成完成: {len(paths)} 个图表 → {output_dir}")
    return paths


# ============================================================================
# Plotly 交互图（可选）
# ============================================================================

def plot_contagion_network_plotly(
    contagion_matrix: pd.DataFrame,
    threshold: float = 0.1,
    output_path: Optional[str] = None,
):
    """
    使用 Plotly 绘制行业风险传导网络图（交互式）。

    Parameters
    ----------
    contagion_matrix : pd.DataFrame
        传导矩阵
    threshold : float
        显示阈值（仅显示传导系数>threshold的边）
    output_path : str, optional
        输出HTML文件路径

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly 未安装，跳过交互式网络图")
        return None

    industries = contagion_matrix.index.tolist()
    n = len(industries)

    # 节点布局（圆形）
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    node_x = np.cos(angles)
    node_y = np.sin(angles)

    # 边
    edge_x, edge_y, edge_weights = [], [], []
    for i in range(n):
        for j in range(n):
            w = contagion_matrix.iloc[i, j]
            if w > threshold:
                edge_x.extend([node_x[i], node_x[j], None])
                edge_y.extend([node_y[i], node_y[j], None])
                edge_weights.append(w)

    fig = go.Figure()

    # 边迹
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none",
        name="传导关系",
    ))

    # 节点迹
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker={"size": 20, "color": "#2E86AB", "line": {"width": 2, "color": "white"}},
        text=industries,
        textposition="top center",
        textfont={"size": 10},
        name="行业",
    ))

    fig.update_layout(
        title="行业 ESG 风险传导网络",
        showlegend=False,
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        width=800, height=800,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"网络图已保存: {output_path}")

    return fig


# ============================================================================
# 便捷函数
# ============================================================================

def save_all_figures_close() -> None:
    """关闭所有 Matplotlib 图形以释放内存。"""
    plt.close("all")


def set_chinese_font(font_path: Optional[str] = None) -> None:
    """
    设置中文字体。

    Parameters
    ----------
    font_path : str, optional
        字体文件路径
    """
    if font_path and os.path.exists(font_path):
        from matplotlib.font_manager import FontProperties
        font = FontProperties(fname=font_path)
        plt.rcParams["font.sans-serif"] = [font.get_name()] + plt.rcParams["font.sans-serif"]
    logger.info(f"中文字体配置: {plt.rcParams['font.sans-serif'][:3]}")
