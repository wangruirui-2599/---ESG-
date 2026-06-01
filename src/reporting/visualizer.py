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
# 行业中英文映射
# ============================================================================

INDUSTRY_NAME_MAP = {
    "银行": "Banking",
    "非银金融": "Non-bank Financial",
    "房地产": "Real Estate",
    "钢铁": "Steel",
    "化工": "Chemical",
    "医药生物": "Pharmaceutical & Bio",
    "电子": "Electronics",
    "计算机": "Computer",
    "食品饮料": "Food & Beverage",
    "汽车": "Automobile",
    "电力设备": "Electrical Equipment",
    "有色金属": "Non-ferrous Metal",
    "采掘": "Mining",
    "公用事业": "Utilities",
    "交通运输": "Transportation",
    "通信": "Communication",
    "机械设备": "Machinery",
    "家用电器": "Household Appliances",
    "国防军工": "Defense & Military",
    "农林牧渔": "Agriculture",
    "商业贸易": "Commerce & Trade",
    "休闲服务": "Leisure Services",
    "纺织服装": "Textile & Apparel",
    "轻工制造": "Light Manufacturing",
    "建筑材料": "Building Materials",
    "建筑装饰": "Building Decoration",
    "电力": "Electric Power",
    "家电": "Home Appliances",
    "信息技术": "Information Technology",
    "采矿": "Mining",
}


def get_english_industry_name(chinese_name: str) -> str:
    """将中文行业名称转换为英文。"""
    return INDUSTRY_NAME_MAP.get(chinese_name, chinese_name)


# ============================================================================
# 1. ESG 雷达图
# ============================================================================

def _score_label(score: float) -> str:
    """根据分数返回评级标签。"""
    if score >= 80:
        return "优秀"
    elif score >= 70:
        return "良好"
    elif score >= 60:
        return "中等"
    elif score >= 50:
        return "偏低"
    else:
        return "较弱"


def plot_esg_radar(
    e_score: float,
    s_score: float,
    g_score: float,
    esg_total: float = 0.0,
    industry: str = "",
    stock_code: str = "",
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, str]:
    """
    绘制 ESG 三维度雷达图。

    Returns
    -------
    (matplotlib.figure.Figure, str)
        图表对象和分析文本
    """
    categories = ["Environmental (E)", "Social (S)", "Governance (G)"]
    values = [e_score, s_score, g_score]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(30)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    ax.fill(angles, values, color=COLOR_PALETTE["primary"], alpha=0.25)
    ax.plot(angles, values, color=COLOR_PALETTE["primary"], linewidth=2, marker="o", markersize=8)

    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.annotate(f"{value:.0f}", xy=(angle, value),
                    xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    title = f"ESG Score - {get_english_industry_name(industry)}"
    if stock_code:
        title += f" ({stock_code})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"雷达图已保存: {output_path}")

    # ---- 生成具体的中文分析 ----
    dims = {"环境(E)": e_score, "社会(S)": s_score, "治理(G)": g_score}
    best_dim = max(dims, key=dims.get)
    worst_dim = min(dims, key=dims.get)
    dim_labels = {k: _score_label(v) for k, v in dims.items()}
    spread = max(dims.values()) - min(dims.values())

    lines = [
        f"ESG综合得分{esg_total:.0f}分，处于{_score_label(esg_total)}水平。",
        f"环境(E)维度{e_score:.0f}分（{dim_labels['环境(E)']}），"
        f"社会(S)维度{s_score:.0f}分（{dim_labels['社会(S)']}），"
        f"治理(G)维度{g_score:.0f}分（{dim_labels['治理(G)']}）。",
    ]

    if spread >= 15:
        lines.append(
            f"三维度分化明显，{best_dim}得分最高（{dims[best_dim]:.0f}分），"
            f"{worst_dim}得分最低（{dims[worst_dim]:.0f}分），差距达{spread:.0f}分，"
            f"建议重点关注{worst_dim}的改善。"
        )
    else:
        lines.append(f"三维度发展较为均衡（极差仅{spread:.0f}分），ESG治理结构相对完善。")

    # 具体建议
    if e_score < 60:
        lines.append(f"环境得分偏低（{e_score:.0f}分），建议加强碳排放管理和清洁能源使用比例。")
    if s_score < 60:
        lines.append(f"社会得分偏低（{s_score:.0f}分），建议关注员工福利和供应链社会责任。")
    if g_score < 60:
        lines.append(f"治理得分偏低（{g_score:.0f}分），建议提升董事会独立性和信息披露透明度。")

    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


# ============================================================================
# 2. 行业权重热力图
# ============================================================================

def plot_industry_weight_heatmap(
    weight_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, str]:
    """
    绘制行业 ESG 权重热力图。

    Returns
    -------
    (matplotlib.figure.Figure, str)
    """
    fig, ax = plt.subplots(figsize=(10, max(8, len(weight_df) * 0.35)))

    heatmap_data = weight_df.set_index("industry")[["E_weight", "S_weight", "G_weight"]]

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=0.6, linewidths=0.5, cbar_kws={"label": "Weight"}, ax=ax)

    ax.set_title("Industry ESG Weight Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("ESG Dimension", fontsize=11)
    ax.set_ylabel("Industry", fontsize=11)

    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"热力图已保存: {output_path}")

    # ---- 分析文本 ----
    lines = []
    for _, row in weight_df.iterrows():
        ind = row.get("industry", "?")
        ew, sw, gw = row.get("E_weight", 0), row.get("S_weight", 0), row.get("G_weight", 0)
        max_w = max(ew, sw, gw)
        focus = "环境(E)" if max_w == ew else "社会(S)" if max_w == sw else "治理(G)"
        lines.append(f"{ind}：{focus}权重最高（{max_w:.0%}），"
                     f"反映该行业ESG核心关切在{focus}维度。")
    analysis_text = "\n".join(lines)
    return fig, analysis_text


# ============================================================================
# 3. DCF 情景估值瀑布图
# ============================================================================

def plot_dcf_scenario_waterfall(
    scenario_results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, str]:
    """
    绘制 DCF 多情景估值对比图。

    Returns
    -------
    (matplotlib.figure.Figure, str)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scenarios = scenario_results.get("scenarios", [])
    names = [s["name"] for s in scenarios]
    ivalues = [s["intrinsic_value"] for s in scenarios]
    probs = [s["probability"] for s in scenarios]
    current_price = scenario_results.get("current_price", 0)
    expected_value = scenario_results.get("expected_value", 0)

    colors = [COLOR_PALETTE["green"], COLOR_PALETTE["blue"], COLOR_PALETTE["red"]]
    axes[0].bar(names, ivalues, color=colors[:len(names)], alpha=0.8, edgecolor="white")
    axes[0].axhline(y=current_price, color="black", linestyle="--", linewidth=2,
                    label=f"Current Price {current_price:.2f}")
    axes[0].axhline(y=expected_value, color=COLOR_PALETTE["primary"], linestyle="-", linewidth=2,
                    label=f"Expected Value {expected_value:.2f}")

    for i, (v, p) in enumerate(zip(ivalues, probs)):
        axes[0].annotate(f"P={p:.0%}", (i, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=10)

    axes[0].set_title("DCF Multi-Scenario Valuation", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Intrinsic Value per Share (CNY)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)

    wedges, texts, autotexts = axes[1].pie(
        probs, labels=names, autopct="%1.1f%%",
        colors=colors[:len(names)], startangle=90, explode=(0.05, 0, 0.05))
    for at in autotexts:
        at.set_fontweight("bold")
    axes[1].set_title("Scenario Probability Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"瀑布图已保存: {output_path}")

    # ---- 分析文本 ----
    upside = (expected_value - current_price) / current_price * 100 if current_price > 0 else 0
    opt_val = ivalues[0] if len(ivalues) > 0 else 0
    pes_val = ivalues[-1] if len(ivalues) > 1 else 0
    valuation_gap = (opt_val - pes_val) / expected_value * 100 if expected_value > 0 else 0

    lines = [
        f"当前股价{current_price:.2f}元，概率加权期望估值{expected_value:.2f}元，"
        f"{'上行空间' if upside >= 0 else '下行风险'}{abs(upside):.1f}%。",
        f"乐观情景（概率{probs[0]:.0%}）估值{opt_val:.2f}元，"
        f"悲观情景（概率{probs[-1]:.0%}）估值{pes_val:.2f}元，"
        f"情景间估值跨度{valuation_gap:.0f}%，{'不确定性较高' if valuation_gap > 50 else '估值区间合理'}。",
    ]
    if upside > 15:
        lines.append(f"Current Price显著低于期望估值，安全边际充足，建议关注买入机会。")
    elif upside > 0:
        lines.append(f"Current Price略低于期望估值，存在一定上行空间，建议逢低布局。")
    elif upside > -10:
        lines.append(f"Current Price接近期望估值，估值基本合理，建议持有观望。")
    else:
        lines.append(f"Current Price高于期望估值，估值偏高，建议谨慎追高。")

    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


# ============================================================================
# 4. 四因子贡献饼图
# ============================================================================

def plot_factor_contribution(
    fusion_result: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, str]:
    """
    绘制四因子贡献可视化。

    Returns
    -------
    (matplotlib.figure.Figure, str)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    final_val = fusion_result.get("final_value", 0)
    axes[0].axhline(y=final_val, color="black", linestyle="--", linewidth=2,
                    label=f"Composite Value: {final_val:.2f}")

    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                     f"{val:.2f}", ha="center", fontsize=10)

    axes[0].set_title("Four-Factor Contribution Breakdown", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Value Contribution (CNY)", fontsize=11)
    axes[0].legend(fontsize=9)

    pcts = [
        fusion_result.get("dcf_contrib_pct", 25),
        fusion_result.get("relative_contrib_pct", 25),
        fusion_result.get("esg_contrib_pct", 25),
        fusion_result.get("sentiment_contrib_pct", 25),
    ]
    axes[1].pie(pcts, labels=names, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title("Factor Contribution Share", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"因子贡献图已保存: {output_path}")

    # ---- 分析文本 ----
    max_factor = names[values.index(max(values))] if max(values) > 0 else "N/A"
    total_contrib = sum(v for v in values if v > 0)
    esg_contrib = contribs["ESG Factor"]
    lines = [
        f"综合估值{final_val:.2f}元，主要由{max_factor}驱动。",
        f"ESG因子贡献{esg_contrib:.2f}元（占比{pcts[2]:.1f}%），"
        f"{'ESG溢价显著，反映市场对其可持续发展能力的认可' if esg_contrib > total_contrib * 0.25 else 'ESG因子影响适中' if esg_contrib > 0 else 'ESG因子贡献为负，可能制约估值提升'}。",
    ]
    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


# ============================================================================
# 5. 异常概率分布直方图
# ============================================================================

def plot_anomaly_distribution(
    anomaly_probs: pd.Series,
    risk_levels: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, str]:
    """
    绘制异常概率分布直方图。

    Returns
    -------
    (matplotlib.figure.Figure, str)
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    n, bins, patches = ax.hist(
        anomaly_probs, bins=30, alpha=0.7,
        color=COLOR_PALETTE["secondary"], edgecolor="white")

    ax.axvspan(0, 0.2, alpha=0.1, color="green", label="Low Risk (<0.2)")
    ax.axvspan(0.2, 0.4, alpha=0.1, color="yellow", label="Moderate-Low Risk (0.2-0.4)")
    ax.axvspan(0.4, 0.6, alpha=0.1, color="orange", label="Medium Risk (0.4-0.6)")
    ax.axvspan(0.6, 0.8, alpha=0.1, color="orangered", label="Moderate-High Risk (0.6-0.8)")
    ax.axvspan(0.8, 1.0, alpha=0.1, color="red", label="High Risk (>0.8)")

    mean_val = anomaly_probs.mean()
    median_val = anomaly_probs.median()
    ax.axvline(x=mean_val, color="black", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.3f}")
    ax.axvline(x=median_val, color="gray", linestyle=":", linewidth=2,
               label=f"Median: {median_val:.3f}")

    ax.set_xlabel("Anomaly Probability", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Financial Anomaly Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"异常分布图已保存: {output_path}")

    # ---- 分析文本 ----
    total = len(anomaly_probs)
    high_risk = (anomaly_probs > 0.6).sum()
    med_risk = ((anomaly_probs > 0.3) & (anomaly_probs <= 0.6)).sum()
    low_risk = (anomaly_probs <= 0.3).sum()
    max_prob = anomaly_probs.max()

    lines = [
        f"共分析{total}个标的，异常概率均值{mean_val:.1%}，中位数{median_val:.1%}。",
        f"低风险标的{low_risk}个（{low_risk/total:.0%}），"
        f"中等风险{med_risk}个（{med_risk/total:.0%}），"
        f"高风险{high_risk}个（{high_risk/total:.0%}）。",
    ]
    if high_risk > total * 0.3:
        lines.append(f"高风险标的占比超过30%，整体财务质量需警惕，建议逐项排查高异常概率标的。")
    elif max_prob > 0.8:
        lines.append(f"存在极端异常标的（最高概率{max_prob:.0%}），建议重点核查该标的财务数据真实性。")
    else:
        lines.append(f"整体异常风险可控，多数标的处于低风险区间。")

    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


# ============================================================================
# 6. 组合净值曲线
# ============================================================================

def plot_portfolio_nav(
    nav_data: pd.DataFrame,
    benchmark_nav: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, str]:
    """
    绘制组合净值曲线与回撤。

    Returns
    -------
    (matplotlib.figure.Figure, str)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})

    axes[0].plot(nav_data["day"], nav_data["nav"],
                 color=COLOR_PALETTE["primary"], linewidth=2, label="Strategy Portfolio")
    if benchmark_nav is not None:
        axes[0].plot(benchmark_nav["day"], benchmark_nav["nav"],
                     color=COLOR_PALETTE["gray"], linewidth=1.5, linestyle="--",
                     label="Benchmark Index")

    axes[0].axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    axes[0].fill_between(nav_data["day"], 1, nav_data["nav"],
                         where=(nav_data["nav"] >= 1),
                         color=COLOR_PALETTE["green"], alpha=0.15)
    axes[0].fill_between(nav_data["day"], 1, nav_data["nav"],
                         where=(nav_data["nav"] < 1),
                         color=COLOR_PALETTE["red"], alpha=0.15)
    axes[0].set_title("Portfolio NAV Curve", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("NAV", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=10)

    nav = nav_data["nav"].values
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak * 100

    axes[1].fill_between(nav_data["day"], 0, drawdown,
                         color=COLOR_PALETTE["red"], alpha=0.4)
    axes[1].plot(nav_data["day"], drawdown,
                 color=COLOR_PALETTE["red"], linewidth=1.5)
    axes[1].set_ylabel("Drawdown (%)", fontsize=11)
    axes[1].set_xlabel("Trading Day", fontsize=11)
    axes[1].axhline(y=0, color="black", linestyle=":", linewidth=0.5)

    max_dd_idx = np.argmin(drawdown)
    axes[1].annotate(
        f"Max Drawdown: {drawdown[max_dd_idx]:.1f}%",
        xy=(max_dd_idx, drawdown[max_dd_idx]),
        xytext=(max_dd_idx + 10, drawdown[max_dd_idx] - 5),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"净值曲线已保存: {output_path}")

    # ---- 分析文本 ----
    final_nav = nav[-1] if len(nav) > 0 else 1.0
    total_return = (final_nav - 1.0) * 100
    max_dd = abs(drawdown[min(len(drawdown)-1, max_dd_idx)]) if len(drawdown) > 0 else 0

    lines = [
        f"策略期末净值{final_nav:.3f}，累计收益{total_return:+.1f}%，"
        f"最大回撤{max_dd:.1f}%。",
    ]
    if total_return > 0 and max_dd < 15:
        lines.append(f"收益表现稳健，回撤控制良好，风险调整后收益较为理想。")
    elif max_dd > 25:
        lines.append(f"最大回撤超过25%，策略波动较大，建议优化风控参数以降低尾部风险。")
    else:
        lines.append(f"收益与回撤处于合理区间，需关注市场环境变化对策略稳定性的影响。")

    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


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
) -> Tuple[plt.Figure, str]:
    """
    绘制 ESG 趋势-动量散点图。

    Returns
    -------
    (matplotlib.figure.Figure, str)
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
        ax.scatter(subset[x_col], subset[y_col],
                   c=color, label=label, alpha=0.6, s=60, edgecolors="white")

    if len(df) > 0:
        top_n = min(5, len(df))
        top_improvers = df.nlargest(top_n, y_col)
        top_decliners = df.nsmallest(top_n, y_col)
        for _, row in pd.concat([top_improvers, top_decliners]).iterrows():
            ax.annotate(str(row.get(stock_col, "")),
                        (row[x_col], row[y_col]),
                        textcoords="offset points",
                        xytext=(5, 5), fontsize=8, alpha=0.8)

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("ESG Momentum Score", fontsize=11)
    ax.set_ylabel("ESG Trend Score", fontsize=11)
    ax.set_title("ESG Trend Analysis Scatter", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info(f"趋势散点图已保存: {output_path}")

    # ---- 分析文本 ----
    trend_counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
    improving = trend_counts.get("显著改善", 0) + trend_counts.get("改善", 0)
    declining = trend_counts.get("显著恶化", 0) + trend_counts.get("恶化", 0)
    stable = trend_counts.get("稳定", 0)
    positive_momentum = (df[x_col] > 0).sum() if x_col in df.columns else 0

    lines = [
        f"共追踪{len(df)}条ESG记录：趋势改善{improving}条，"
        f"恶化{declining}条，稳定{stable}条。",
        f"正向动量标的{positive_momentum}个（{positive_momentum/len(df):.0%}），"
        f"{'ESG整体向好，多数标的在持续改善' if positive_momentum > len(df)*0.6 else 'ESG趋势分化明显，需区分对待' if positive_momentum > len(df)*0.3 else 'ESG整体承压，多数标的动量偏弱'}。",
    ]
    if declining > improving:
        lines.append(f"恶化标的数超过改善标的数，建议回避趋势持续恶化的标的，关注底部反转信号。")

    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


# ============================================================================
# 8. 综合仪表板
# ============================================================================

def plot_dashboard(
    esg_radar_data: Dict[str, float],
    scenario_data: Dict[str, Any],
    anomaly_data: pd.Series,
    output_dir: str = "output/figures",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    生成综合可视化仪表板（所有图表）。

    Returns
    -------
    (dict, dict)
        (图表文件路径映射, 图表分析文本映射)
    """
    out = _ensure_output_dir(output_dir)
    paths = {}
    analyses = {}

    # 1. ESG雷达图
    paths["radar"] = str(out / "esg_radar.png")
    _, analyses["radar"] = plot_esg_radar(
        esg_radar_data.get("E_score", 0),
        esg_radar_data.get("S_score", 0),
        esg_radar_data.get("G_score", 0),
        esg_radar_data.get("ESG_total", 0),
        output_path=paths["radar"],
    )
    plt.close("all")

    # 2. DCF情景图
    paths["scenario"] = str(out / "dcf_scenario.png")
    _, analyses["scenario"] = plot_dcf_scenario_waterfall(scenario_data, output_path=paths["scenario"])
    plt.close("all")

    # 3. 异常分布图
    paths["anomaly"] = str(out / "anomaly_dist.png")
    _, analyses["anomaly"] = plot_anomaly_distribution(anomaly_data, output_path=paths["anomaly"])
    plt.close("all")

    logger.info(f"仪表板生成完成: {len(paths)} 个图表 → {output_dir}")
    return paths, analyses


# ============================================================================
# Plotly 交互图（可选）
# ============================================================================

def plot_contagion_network_plotly(
    contagion_matrix: pd.DataFrame,
    threshold: float = 0.1,
    output_path: Optional[str] = None,
) -> Optional[Tuple[Any, str]]:
    """
    使用 Plotly 绘制行业风险传导网络图（交互式）。

    Returns
    -------
    (plotly.graph_objects.Figure, str) or None
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly 未安装，跳过交互式网络图")
        return None

    industries = contagion_matrix.index.tolist()
    n = len(industries)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    node_x = np.cos(angles)
    node_y = np.sin(angles)

    edge_x, edge_y, edge_weights = [], [], []
    for i in range(n):
        for j in range(n):
            w = contagion_matrix.iloc[i, j]
            if w > threshold:
                edge_x.extend([node_x[i], node_x[j], None])
                edge_y.extend([node_y[i], node_y[j], None])
                edge_weights.append(w)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none", name="Contagion Link"))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker={"size": 20, "color": "#2E86AB", "line": {"width": 2, "color": "white"}},
        text=industries, textposition="top center", textfont={"size": 10}, name="Industry"))

    fig.update_layout(
        title="Industry ESG Risk Contagion Network",
        showlegend=False,
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        width=800, height=800)

    if output_path:
        fig.write_html(output_path)
        logger.info(f"网络图已保存: {output_path}")

    # ---- 分析文本 ----
    # 找出传导关系最强的边
    max_edge = ("", "", 0)
    for i in range(n):
        for j in range(n):
            w = contagion_matrix.iloc[i, j]
            if w > max_edge[2]:
                max_edge = (industries[i], industries[j], w)

    high_edges = sum(1 for i in range(n) for j in range(n) if contagion_matrix.iloc[i, j] > threshold)
    lines = [
        f"共{len(industries)}个行业节点，发现{high_edges}条显著传导关系（阈值>{threshold:.0%}）。",
    ]
    if max_edge[2] > 0:
        lines.append(
            f"最强传导路径：{max_edge[0]} → {max_edge[1]}（系数{max_edge[2]:.2f}），"
            f"表明{max_edge[0]}行业波动对{max_edge[1]}行业具有显著传导效应。"
        )
    lines.append("建议关注传导密集的枢纽行业，其波动可能引发多行业连锁反应。")

    analysis_text = "\n\n".join(lines)
    return fig, analysis_text


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
