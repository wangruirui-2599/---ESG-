"""
可视化图表生成模块
==================
基于 Matplotlib、Seaborn 和 Plotly 生成专业的金融分析图表。

图表清单：
  1. ESG三维度雷达图
  2. 行业权重热力图
  3. DCF情景估值瀑布图
  4. 四因子贡献饼图/柱状图
  5. Anomaly Probability分布直方图
  6. Portfolio NAV Curve
  7. 风险传导网络图（Plotly）
  8. ESG趋势动量散点图
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

# ---- 在导入 matplotlib 之前先清理字体缓存 ----
# 这是解决中文方框问题的关键：matplotlib 可能在之前的运行中缓存了错误的字体映射
import glob as _glob
import matplotlib as _mpl_pre
_cache_dir = _mpl_pre.get_cachedir()
for _f in _glob.glob(os.path.join(_cache_dir, "fontlist-v*.json")):
    try:
        os.remove(_f)
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # 非交互式后端

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import seaborn as sns
import platform


# ============================================================================
# 中文字体配置（跨平台）
# ============================================================================

# 模块级缓存：setup_chinese_font() 找到的字体文件路径
_CHINESE_FONT_FILE: Optional[str] = None
_CHINESE_FONT_NAME: Optional[str] = None


def setup_chinese_font() -> bool:
    """
    配置 matplotlib 中文字体，解决图表中文显示为方框的问题。

    采用多层策略确保中文正常渲染：
      1. 删除 matplotlib 字体缓存 → 强制重建字体列表
      2. 按操作系统查找中文字体文件
      3. 将字体文件路径保存到模块变量，供各绘图函数使用
      4. 同时设置 rcParams 作为默认回退

    支持的操作系统：
      - Windows: Microsoft YaHei → SimHei → KaiTi
      - macOS:   PingFang SC → Heiti SC → STHeiti
      - Linux:   WenQuanYi Zen Hei → Noto Sans CJK SC

    Returns
    -------
    bool
        True 如果找到并加载了中文字体，False 否则
    """
    global _CHINESE_FONT_FILE, _CHINESE_FONT_NAME

    # ---- 1. 删除旧字体缓存，强制重建 ----
    try:
        cache_dir = matplotlib.get_cachedir()
        for cache_file in Path(cache_dir).glob("fontlist-v*.json"):
            cache_file.unlink(missing_ok=True)
            logger.debug(f"已删除字体缓存: {cache_file}")
    except Exception:
        pass

    # ---- 2. 按操作系统确定候选字体（先于重建，因为需要文件路径） ----
    system = platform.system()
    if system == "Windows":
        font_names = ["Microsoft YaHei", "SimHei", "KaiTi", "FangSong"]
        font_dirs = [Path("C:/Windows/Fonts")]
    elif system == "Darwin":  # macOS
        font_names = ["PingFang SC", "Heiti SC", "STHeiti", "Apple LiSung"]
        font_dirs = [Path("/System/Library/Fonts"),
                      Path("/Library/Fonts")]
    else:  # Linux / other
        font_names = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                       "Noto Sans CJK SC", "Noto Sans SC", "SimHei"]
        font_dirs = [Path("/usr/share/fonts"),
                      Path("/usr/local/share/fonts"),
                      Path.home() / ".fonts"]

    # ---- 3. 查找中文字体文件 ----
    # 先通过 matplotlib 已有列表查找
    available_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
    found_font = None
    found_file = None

    for font_name in font_names:
        if font_name in available_fonts:
            found_font = font_name
            found_file = available_fonts[font_name]
            break

    # 如果按名字找不到，扫描系统字体目录
    if not found_file:
        name_to_file = {
            "Microsoft YaHei": ["msyh.ttc", "msyh.ttf"],
            "SimHei": ["simhei.ttf"],
            "KaiTi": ["simkai.ttf"],
            "FangSong": ["simfang.ttf"],
        }
        for font_name in font_names:
            candidates = name_to_file.get(font_name, [font_name.replace(" ", "") + ".ttf"])
            for font_dir in font_dirs:
                if not font_dir.exists():
                    continue
                for candidate in candidates:
                    for ext in ["", ".ttf", ".ttc", ".otf"]:
                        p = font_dir / (candidate + ext)
                        if p.exists():
                            found_font = font_name
                            found_file = str(p)
                            break
                    if found_file:
                        break
                if found_file:
                    break
            if found_file:
                break

    # ---- 4. 注册字体文件（必须在 _load_fontmanager 之前） ----
    # 注意：只注册 Regular 和 Bold 变体，避免 Light 等细体覆盖 Regular。
    # _load_fontmanager 会以最后注册的同名字体为准。
    if found_file:
        font_dir = Path(found_file).parent
        font_stem = Path(found_file).stem.lower()
        # 去除变体后缀 (Bold/Light/Italic) 获取基础文件名
        for suffix in ["bd", "l", "i", "bi", "z", "lt"]:
            if font_stem.endswith(suffix):
                font_stem = font_stem[:-len(suffix)]
                break
        # 只注册 Regular 和 Bold（跳过 Light 等细体，防止覆盖标准字重）
        skip_suffixes = ("l.ttc", "l.ttf", "lt.ttf", "lt.ttc", "light", "Light")
        for sibling in sorted(font_dir.glob(f"{font_stem}*.*")):
            if sibling.suffix.lower() not in (".ttf", ".ttc", ".otf"):
                continue
            name_lower = sibling.name.lower()
            if any(s in name_lower for s in skip_suffixes):
                logger.debug(f"跳过细体字体: {sibling.name}")
                continue
            try:
                fm.fontManager.addfont(str(sibling))
                logger.debug(f"已注册字体文件: {sibling.name}")
            except Exception:
                pass

    # ---- 5. 重建字体管理器（包含新注册的字体） ----
    # 必须在 addfont 之后调用，让新注册的字体被正确索引
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    # ---- 6. 保存到模块变量 ----
    _CHINESE_FONT_FILE = found_file
    _CHINESE_FONT_NAME = found_font

    # ---- 7. 设置 rcParams ----
    # 关键：font.family 直接设为找到的字体名，避免经 sans-serif 解析链回退到 Arial
    if found_font:
        plt.rcParams["font.family"] = found_font  # 直接指定，绕过 sans-serif 链
        plt.rcParams["font.sans-serif"] = [found_font] + font_names + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(f"中文字体已加载: {found_font} ({found_file})")
        return True
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = font_names + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        logger.warning(
            f"未找到中文字体 (系统: {system})，"
            f"候选: {font_names[:3]}... "
            f"请安装中文字体后删除 {matplotlib.get_cachedir()} 下的缓存文件。"
        )
        return False


# 模块加载时自动配置
# 注意：必须先设 seaborn 样式，再配中文字体！
# 因为 sns.set_style() 会重置 font.family 为 'sans-serif'，覆盖中文字体设置。
sns.set_style("whitegrid")
_font_configured = setup_chinese_font()


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
    categories = ["Environment (E)", "Social (S)", "Governance (G)"]
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

    title = f"ESG Score Radar Chart"
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
        cbar_kws={"label": "Weight"},
        ax=ax,
    )

    ax.set_title("Industry ESG Dimension Weight Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("ESG Dimension", fontsize=11)
    ax.set_ylabel("Industry", fontsize=11)

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
        label=f"Current Price {scenario_results.get('current_price', 0):.2f}",
    )
    axes[0].axhline(
        y=scenario_results.get("expected_value", 0),
        color=COLOR_PALETTE["primary"],
        linestyle="-",
        linewidth=2,
        label=f"Expected Value {scenario_results.get('expected_value', 0):.2f}",
    )

    # 标注概率
    for i, (v, p) in enumerate(zip(ivalues, probs)):
        axes[0].annotate(
            f"P={p:.0%}", (i, v), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=10,
        )

    axes[0].set_title("DCF Multi-Scenario Valuation Comparison", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Intrinsic Value Per Share (CNY)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)

    # 右图：概率饼图
    wedges, texts, autotexts = axes[1].pie(
        probs, labels=names, autopct="%1.1f%%",
        colors=colors[:len(names)],
        startangle=90, explode=(0.05, 0, 0.05),
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[1].set_title("Scenario Probability Distribution", fontsize=13, fontweight="bold")

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
        "DCF Valuation": fusion_result.get("dcf_contrib", 0),
        "Relative Valuation": fusion_result.get("relative_contrib", 0),
        "ESG Factor": fusion_result.get("esg_contrib", 0),
        "Market Sentiment": fusion_result.get("sentiment_contrib", 0),
    }
    names = list(contribs.keys())
    values = list(contribs.values())
    colors = [COLOR_PALETTE["primary"], COLOR_PALETTE["accent"],
              COLOR_PALETTE["green"], COLOR_PALETTE["purple"]]

    bars = axes[0].bar(names, values, color=colors, alpha=0.85, edgecolor="white")
    axes[0].axhline(
        y=fusion_result.get("final_value", 0),
        color="black", linestyle="--", linewidth=2,
        label=f"Composite Valuation:{fusion_result.get('final_value', 0):.2f}",
    )

    # 标注数值
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
            f"{val:.2f}", ha="center", fontsize=10,
        )

    axes[0].set_title("Four-Factor Contribution Breakdown", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Valuation Contribution (CNY)", fontsize=11)
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
    axes[1].set_title("Factor Contribution Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"因子贡献图已保存: {output_path}")

    return fig


# ============================================================================
# 5. Anomaly Probability分布直方图
# ============================================================================

def plot_anomaly_distribution(
    anomaly_probs: pd.Series,
    risk_levels: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制Anomaly Probability分布直方图。

    Parameters
    ----------
    anomaly_probs : pd.Series
        Anomaly Probability序列
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
    ax.axvspan(0, 0.2, alpha=0.1, color="green", label="Low Risk (<0.2)")
    ax.axvspan(0.2, 0.4, alpha=0.1, color="yellow", label="Moderate Risk (0.2-0.4)")
    ax.axvspan(0.4, 0.6, alpha=0.1, color="orange", label="Medium Risk (0.4-0.6)")
    ax.axvspan(0.6, 0.8, alpha=0.1, color="orangered", label="High Risk (0.6-0.8)")
    ax.axvspan(0.8, 1.0, alpha=0.1, color="red", label="Critical Risk (>0.8)")

    ax.axvline(
        x=anomaly_probs.mean(), color="black", linestyle="--", linewidth=2,
        label=f"Mean: {anomaly_probs.mean():.3f}",
    )
    ax.axvline(
        x=anomaly_probs.median(), color="gray", linestyle=":", linewidth=2,
        label=f"Median: {anomaly_probs.median():.3f}",
    )

    ax.set_xlabel("Anomaly Probability", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Financial Anomaly Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"异常分布图已保存: {output_path}")

    return fig


# ============================================================================
# 6. Portfolio NAV Curve
# ============================================================================

def plot_portfolio_nav(
    nav_data: pd.DataFrame,
    benchmark_nav: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制Portfolio NAV Curve与回撤。

    Parameters
    ----------
    nav_data : pd.DataFrame
        组合NAV数据（含 day, nav 列）
    benchmark_nav : pd.DataFrame, optional
        基准NAV数据
    output_path : str, optional
        保存路径

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})

    # 上图：NAV曲线
    axes[0].plot(
        nav_data["day"], nav_data["nav"],
        color=COLOR_PALETTE["primary"], linewidth=2, label="Strategy Portfolio",
    )
    if benchmark_nav is not None:
        axes[0].plot(
            benchmark_nav["day"], benchmark_nav["nav"],
            color=COLOR_PALETTE["gray"], linewidth=1.5, linestyle="--",
            label="Benchmark Index",
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
    axes[0].set_title("Portfolio NAV Curve", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("NAV", fontsize=11)
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
    axes[1].set_ylabel("Drawdown (%)", fontsize=11)
    axes[1].set_xlabel("Trading Days", fontsize=11)
    axes[1].axhline(y=0, color="black", linestyle=":", linewidth=0.5)

    # 最大回撤标注
    max_dd_idx = np.argmin(drawdown)
    axes[1].annotate(
        f"Max Drawdown:{drawdown[max_dd_idx]:.1f}%",
        xy=(max_dd_idx, drawdown[max_dd_idx]),
        xytext=(max_dd_idx + 10, drawdown[max_dd_idx] - 5),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10, fontweight="bold",
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"NAV曲线已保存: {output_path}")

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
        "Significant Improvement": COLOR_PALETTE["green"],
        "Improvement": "#27AE60",
        "Stable": COLOR_PALETTE["gray"],
        "Deterioration": COLOR_PALETTE["orange"],
        "Significant Deterioration": COLOR_PALETTE["red"],
        "Volatility": COLOR_PALETTE["purple"],
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
    ax.set_xlabel("ESG Momentum Score", fontsize=11)
    ax.set_ylabel("ESG Trend Score", fontsize=11)
    ax.set_title("ESG Trend Analysis Scatter Plot", fontsize=14, fontweight="bold")
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
        Anomaly Probability序列
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
        title="Industry ESG Risk Contagion Network",
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
    设置中文字体（兼容旧接口）。

    优先使用模块加载时自动配置的字体。
    如需使用自定义字体文件，传入 font_path 参数。

    Parameters
    ----------
    font_path : str, optional
        自定义字体文件路径。如不传则使用 setup_chinese_font() 的自动检测结果。
    """
    if font_path and os.path.exists(font_path):
        # 使用指定的字体文件
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
        # 重建字体缓存
        fm._load_fontmanager(try_read_cache=False)
        logger.info(f"自定义字体已加载: {font_name} ({font_path})")
    else:
        # 回退到自动检测
        if not _font_configured:
            setup_chinese_font()
    logger.info(f"当前字体配置: {plt.rcParams['font.sans-serif'][:4]}")
