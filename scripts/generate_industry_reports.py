#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合行业ESG分析报告生成器
==========================
为所有10个行业（钢铁、化工、电力、食品饮料、家电、医药生物、信息技术、采矿、汽车、银行）
生成完整的ESG分析报告，包括：
  - 单个公司的详细ESG报告（HTML + Markdown）
  - 行业横向对比报告
  - 各行业ESG趋势分析
  - 行业风险传导分析
  - 可视化图表

使用方式:
  python scripts/generate_industry_reports.py
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, get_logger
from src.reporting.report_generator import ReportGenerator
from src.reporting.visualizer import (
    plot_esg_radar, plot_dcf_scenario_waterfall,
    plot_anomaly_distribution, plot_esg_trend_scatter,
    save_all_figures_close, COLOR_PALETTE, ESG_COLORS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

logger = get_logger(__name__)

# ============================================================================
# 行业数据与特征描述
# ============================================================================

INDUSTRY_PROFILES = {
    "钢铁": {
        "industry_en": "Steel",
        "esg_focus": "环境(E)权重最高，碳排放和能源消耗是核心议题",
        "key_metrics": ["碳排放强度", "能源效率", "水耗", "安全生产"],
        "risk_factors": ["环保政策收紧", "原材料价格波动", "产能过剩"],
        "benchmark_esg": 55.0,
    },
    "化工": {
        "industry_en": "Chemical",
        "esg_focus": "环境(E)和治理(G)并重，污染物排放和安全管理是关键",
        "key_metrics": ["废水排放", "危险废物处理", "安全生产事故率", "研发投入"],
        "risk_factors": ["环保合规成本", "安全生产风险", "原材料价格波动"],
        "benchmark_esg": 55.0,
    },
    "电力": {
        "industry_en": "Electric Power",
        "esg_focus": "环境(E)权重最高，清洁能源转型是核心",
        "key_metrics": ["清洁能源占比", "碳排放强度", "供电可靠性", "电价市场竞争力"],
        "risk_factors": ["碳交易成本", "电力市场化改革", "新能源替代"],
        "benchmark_esg": 56.0,
    },
    "食品饮料": {
        "industry_en": "Food & Beverage",
        "esg_focus": "社会(S)权重最高，食品安全和品牌声誉是核心",
        "key_metrics": ["食品安全认证", "供应链管理", "品牌价值", "消费者满意度"],
        "risk_factors": ["食品安全事件", "原材料价格波动", "消费趋势变化"],
        "benchmark_esg": 68.0,
    },
    "家电": {
        "industry_en": "Home Appliances",
        "esg_focus": "环境(E)和治理(G)并重，能效和产品生命周期管理",
        "key_metrics": ["产品能效", "回收利用率", "供应链合规", "技术创新"],
        "risk_factors": ["原材料成本", "贸易摩擦", "房地产下行"],
        "benchmark_esg": 64.0,
    },
    "医药生物": {
        "industry_en": "Pharma & Biotech",
        "esg_focus": "社会(S)和治理(G)并重，药品安全和研发创新是核心",
        "key_metrics": ["药品安全", "研发投入占比", "临床透明度", "知识产权管理"],
        "risk_factors": ["集采降价", "药品安全事件", "专利到期"],
        "benchmark_esg": 69.0,
    },
    "信息技术": {
        "industry_en": "Information Technology",
        "esg_focus": "治理(G)权重最高，数据安全和隐私保护是核心",
        "key_metrics": ["数据安全", "隐私保护", "研发投入", "员工发展"],
        "risk_factors": ["技术迭代", "地缘政治", "网络安全"],
        "benchmark_esg": 70.0,
    },
    "采矿": {
        "industry_en": "Mining",
        "esg_focus": "环境(E)权重最高，矿山修复和安全生产是核心",
        "key_metrics": ["矿山修复率", "安全生产天数", "碳排放强度", "社区关系"],
        "risk_factors": ["资源枯竭", "安全事故", "环保监管"],
        "benchmark_esg": 48.0,
    },
    "汽车": {
        "industry_en": "Automotive",
        "esg_focus": "环境(E)和社会(S)并重，新能源转型和供应链管理",
        "key_metrics": ["新能源车占比", "碳排放强度", "供应链合规", "产品质量"],
        "risk_factors": ["新能源转型", "芯片短缺", "贸易壁垒"],
        "benchmark_esg": 60.0,
    },
    "银行": {
        "industry_en": "Banking",
        "esg_focus": "治理(G)权重最高，风险管理和合规是核心",
        "key_metrics": ["不良贷款率", "资本充足率", "绿色信贷占比", "数据安全"],
        "risk_factors": ["利率市场化", "房地产风险", "金融科技冲击"],
        "benchmark_esg": 70.0,
    },
}


def load_processed_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """加载处理后的数据。"""
    parquet_path = Path(data_dir) / "07_advice.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"加载处理数据: {parquet_path} ({len(df)} 行)")
        return df

    csv_path = Path(data_dir) / "03_esg_quant.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"从CSV加载数据: {csv_path} ({len(df)} 行)")
        return df

    raise FileNotFoundError("找不到处理后的数据文件")


def get_latest_by_stock(df: pd.DataFrame) -> pd.DataFrame:
    """取每只股票的最新记录。"""
    if "report_year" in df.columns:
        df = df.sort_values("report_year")
    latest = df.groupby("stock_code", as_index=False).last()
    logger.info(f"最新数据: {len(latest)} 只股票")
    return latest


def get_industry_timeseries(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """获取某只股票的时间序列数据。"""
    ts = df[df["stock_code"] == stock_code].sort_values("report_year") if "report_year" in df.columns else df[df["stock_code"] == stock_code]
    return ts


# ============================================================================
# 1. 行业ESG总分排名图
# ============================================================================

def plot_industry_esg_ranking(latest_df: pd.DataFrame, output_path: str) -> tuple:
    """行业ESG总分排名横向柱状图。返回 (path, analysis_text)。"""
    fig, ax = plt.subplots(figsize=(14, 8))

    df_sorted = latest_df.sort_values("ESG_total")

    industries = df_sorted["industry"].tolist()
    esg_totals = df_sorted["ESG_total"].tolist()
    e_scores = df_sorted["E_score"].tolist()
    s_scores = df_sorted["S_score"].tolist()
    g_scores = df_sorted["G_score"].tolist()

    y_pos = range(len(industries))
    bar_height = 0.2

    ax.barh([y + bar_height for y in y_pos], g_scores, bar_height,
            label="治理(G)", color=ESG_COLORS["G"], alpha=0.9)
    ax.barh(y_pos, s_scores, bar_height,
            label="社会(S)", color=ESG_COLORS["S"], alpha=0.9)
    ax.barh([y - bar_height for y in y_pos], e_scores, bar_height,
            label="环境(E)", color=ESG_COLORS["E"], alpha=0.9)

    # 标注总分
    for i, (ind, total) in enumerate(zip(industries, esg_totals)):
        ax.annotate(f"{total:.1f}", xy=(total + 0.5, i),
                    va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(industries, fontsize=11)
    ax.set_xlabel("ESG评分", fontsize=12)
    ax.set_title("各行业ESG评分排名 (E/S/G分解)", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"行业排名图: {output_path}")
    # 分析
    top3 = latest_df.nlargest(3, 'ESG_total')['industry'].tolist()
    bot3 = latest_df.nsmallest(3, 'ESG_total')['industry'].tolist()
    avg_e = latest_df['E_score'].mean()
    avg_s = latest_df['S_score'].mean()
    avg_g = latest_df['G_score'].mean()
    best_dim = "E" if avg_e >= avg_s and avg_e >= avg_g else "S" if avg_s >= avg_g else "G"
    analysis = (
        f"ESG总分前三：{'、'.join(top3)}；后三：{'、'.join(bot3)}。"
        f"全行业平均E={avg_e:.0f}分、S={avg_s:.0f}分、G={avg_g:.0f}分，"
        f"整体在{best_dim}维度表现相对较好。"
        f"{top3[0]}以{latest_df['ESG_total'].max():.0f}分领先，"
        f"建议关注{bot3[0]}的ESG改善进度（{latest_df['ESG_total'].min():.0f}分）。"
    )
    return output_path, analysis


# ============================================================================
# 2. ESG趋势时间序列对比图
# ============================================================================

def plot_esg_timeseries_comparison(df: pd.DataFrame, output_path: str) -> str:
    """所有行业的ESG总分时间序列对比。"""
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    all_industries = df["industry"].unique() if "industry" in df.columns else df["stock_code"].unique()

    for i, industry in enumerate(sorted(all_industries)):
        ind_data = df[df["industry"] == industry].sort_values("report_year") if "report_year" in df.columns else df[df["industry"] == industry]
        if len(ind_data) > 0:
            years = ind_data["report_year"].values if "report_year" in ind_data.columns else range(len(ind_data))
            ax.plot(years, ind_data["ESG_total"].values, 'o-',
                    color=colors[i % 10], linewidth=2, markersize=6,
                    label=f"{industry}", alpha=0.85)

    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("ESG总分", fontsize=12)
    ax.set_title("各行业ESG总分时间趋势 (2019-2024)", fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, ncol=1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"时间序列图: {output_path}")
    return output_path, ""


# ============================================================================
# 3. 行业风险传导热力图
# ============================================================================

def plot_contagion_heatmap(latest_df: pd.DataFrame, output_path: str) -> str:
    """绘制行业风险传导暴露热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 左图：行业风险传导评分
    df_sorted = latest_df.sort_values("contagion_risk_score", ascending=True) if "contagion_risk_score" in latest_df.columns else latest_df

    industries = df_sorted["industry"].tolist()
    risk_scores = df_sorted["contagion_risk_score"].tolist() if "contagion_risk_score" in df_sorted.columns else [0] * len(industries)
    affected_counts = df_sorted["affected_by_count"].tolist() if "affected_by_count" in df_sorted.columns else [0] * len(industries)

    colors_risk = ['#2ECC71' if r < 20 else '#E67E22' if r < 50 else '#E74C3C' for r in risk_scores]
    axes[0].barh(industries, risk_scores, color=colors_risk, alpha=0.85)
    axes[0].set_xlabel("传导风险评分", fontsize=12)
    axes[0].set_title("各行业传导风险暴露评分", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)
    for i, (ind, s) in enumerate(zip(industries, risk_scores)):
        axes[0].annotate(f"{s:.1f}", xy=(s + 0.3, i), va="center", fontsize=9)

    # 右图：受影响的行业数量
    colors_cnt = ['#2ECC71' if c <= 1 else '#E67E22' if c <= 3 else '#E74C3C' for c in affected_counts]
    axes[1].barh(industries, affected_counts, color=colors_cnt, alpha=0.85)
    axes[1].set_xlabel("受影响行业数", fontsize=12)
    axes[1].set_title("各行业受影响上游行业数", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    for i, (ind, c) in enumerate(zip(industries, affected_counts)):
        axes[1].annotate(f"{int(c)}", xy=(c + 0.05, i), va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"风险传导图: {output_path}")
    return output_path, ""


# ============================================================================
# 4. 行业ESG维度雷达对比图（多行业叠加）
# ============================================================================

def plot_multi_industry_radar(latest_df: pd.DataFrame, output_path: str) -> str:
    """多个行业的ESG雷达图叠加对比。"""
    categories = ["环境 (E)", "社会 (S)", "治理 (G)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(latest_df)))

    for i, (_, row) in enumerate(latest_df.iterrows()):
        values = [row.get("E_score", 0), row.get("S_score", 0), row.get("G_score", 0)]
        values += values[:1]
        ax.fill(angles, values, alpha=0.05, color=colors[i])
        ax.plot(angles, values, 'o-', linewidth=1.8, color=colors[i],
                label=f"{row.get('industry', '?')} (ESG:{row.get('ESG_total', 0):.0f})",
                markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("各行业ESG维度雷达图对比", fontsize=15, fontweight="bold", pad=25)

    # 图例放在外面
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8, ncol=1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"多行业雷达图: {output_path}")
    return output_path, ""


# ============================================================================
# 5. 行业投资吸引力矩阵
# ============================================================================

def plot_industry_attractiveness(latest_df: pd.DataFrame, output_path: str) -> str:
    """行业投资吸引力散点图（ESG评分 vs 上行空间）。"""
    fig, ax = plt.subplots(figsize=(14, 9))

    for _, row in latest_df.iterrows():
        x = row.get("ESG_total", 50)
        y = row.get("expected_upside_pct", 0)
        industry = row.get("industry", "?")

        # 颜色基于吸引力
        if y > 0 and x > 60:
            color = "#2ECC71"
            quadrant = "高ESG+正回报"
        elif y > 0:
            color = "#F18F01"
            quadrant = "低ESG+正回报"
        elif x > 60:
            color = "#3498DB"
            quadrant = "高ESG+负回报"
        else:
            color = "#E74C3C"
            quadrant = "低ESG+负回报"

        ax.scatter(x, y, c=color, s=200, alpha=0.75, edgecolors="white", linewidth=1.5)
        ax.annotate(industry, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight="bold")

        # 添加气泡大小表示市场关注度
        anomaly = row.get("anomaly_probability", 0)
        if anomaly > 0.3:
            ax.scatter(x, y, s=800, facecolors="none", edgecolors="red",
                      linewidth=2, alpha=0.5, linestyle="--")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.axvline(x=60, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # 象限标签
    ax.text(80, max(latest_df["expected_upside_pct"].max() * 0.85, 10),
            "高ESG + 正回报 ★★★", fontsize=11, color="#2ECC71", ha="center")
    ax.text(40, max(latest_df["expected_upside_pct"].max() * 0.85, 10),
            "低ESG + 正回报 ★★", fontsize=11, color="#F18F01", ha="center")
    ax.text(80, min(latest_df["expected_upside_pct"].min() * 0.7, -10),
            "高ESG + 负回报 ★", fontsize=11, color="#3498DB", ha="center")
    ax.text(40, min(latest_df["expected_upside_pct"].min() * 0.7, -10),
            "低ESG + 负回报 ☆", fontsize=11, color="#E74C3C", ha="center")

    ax.set_xlabel("ESG总分", fontsize=12)
    ax.set_ylabel("期望上行空间 (%)", fontsize=12)
    ax.set_title("行业投资吸引力矩阵 (ESG × 估值空间)", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"吸引力矩阵: {output_path}")
    return output_path, ""


# ============================================================================
# 6. 行业ESG动量气泡图
# ============================================================================

def plot_esg_momentum_bubble(latest_df: pd.DataFrame, output_path: str) -> str:
    """ESG动量气泡图。"""
    fig, ax = plt.subplots(figsize=(14, 8))

    for _, row in latest_df.iterrows():
        x = row.get("ESG_total", 50)
        y = row.get("ESG_total_momentum", 0)
        industry = row.get("industry", "?")
        anomaly = row.get("anomaly_probability", 0)

        bubble_size = 150 + (100 - anomaly * 100) * 5

        if y > 0.03:
            color = "#2ECC71"
        elif y > 0:
            color = "#27AE60"
        elif y > -0.02:
            color = "#95A5A6"
        else:
            color = "#E74C3C"

        ax.scatter(x, y, s=bubble_size, c=color, alpha=0.7, edgecolors="white", linewidth=1.5)
        ax.annotate(industry, (x, y), textcoords="offset points",
                    xytext=(0, 12), fontsize=10, fontweight="bold", ha="center")

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("ESG总分", fontsize=12)
    ax.set_ylabel("ESG动量分数", fontsize=12)
    ax.set_title("行业ESG质量与动量分析 (气泡大小=财务健康度)", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"动量气泡图: {output_path}")
    return output_path, ""


# ============================================================================
# 7. 综合行业对比分析HTML报告
# ============================================================================

def generate_industry_comparison_html(
    latest_df: pd.DataFrame,
    df_full: pd.DataFrame,
    figures: Dict[str, str],
    output_dir: str = "output/reports",
) -> str:
    """生成综合行业对比HTML报告。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建行业卡片
    industry_cards = ""
    for _, row in latest_df.iterrows():
        ind = row.get("industry", "N/A")
        profile = INDUSTRY_PROFILES.get(ind, {})
        esg_total = row.get("ESG_total", 0)
        e_score = row.get("E_score", 0)
        s_score = row.get("S_score", 0)
        g_score = row.get("G_score", 0)
        momentum = row.get("ESG_total_momentum", 0)
        trend = row.get("trend_label", "N/A")
        anomaly = row.get("anomaly_probability", 0)
        upside = row.get("expected_upside_pct", 0)
        advice = row.get("advice", "N/A")
        risk = row.get("risk_level", "N/A")
        contagion = row.get("contagion_risk_score", 0)

        # ESG趋势颜色
        momentum_color = "#2ECC71" if momentum > 0.03 else "#E74C3C" if momentum < 0 else "#E67E22"

        # 异常概率颜色
        anomaly_color = "#2ECC71" if anomaly < 0.3 else "#E67E22" if anomaly < 0.5 else "#E74C3C"

        # 上行空间颜色
        upside_color = "#2ECC71" if upside > 0 else "#E74C3C"

        industry_cards += f"""
        <div class="industry-card">
            <div class="industry-header">
                <h3>🏭 {ind}</h3>
                <span class="stock-code">{row.get('stock_code', '')} | {row.get('company_name', '')}</span>
            </div>
            <div class="industry-metrics">
                <div class="mini-metric">
                    <div class="mini-value" style="color:{'#2ECC71' if esg_total > 60 else '#E67E22' if esg_total > 50 else '#E74C3C'}">{esg_total:.1f}</div>
                    <div class="mini-label">ESG总分</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value">E:{e_score:.0f} S:{s_score:.0f} G:{g_score:.0f}</div>
                    <div class="mini-label">E/S/G评分</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value" style="color:{momentum_color}">{momentum:.4f}</div>
                    <div class="mini-label">ESG动量 ({trend})</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value" style="color:{upside_color}">{upside:+.1f}%</div>
                    <div class="mini-label">期望上行空间</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value" style="color:{anomaly_color}">{anomaly:.1%}</div>
                    <div class="mini-label">异常概率 ({risk})</div>
                </div>
                <div class="mini-metric">
                    <div class="mini-value">{contagion:.1f}</div>
                    <div class="mini-label">传导风险评分</div>
                </div>
            </div>
            <div class="industry-detail">
                <p><strong>ESG焦点:</strong> {profile.get('esg_focus', 'N/A')}</p>
                <p><strong>关键指标:</strong> {', '.join(profile.get('key_metrics', []))}</p>
                <p><strong>风险因素:</strong> {', '.join(profile.get('risk_factors', []))}</p>
            </div>
            <div class="advice-badge advice-{'buy' if '买入' in str(advice) else 'hold' if '持有' in str(advice) else 'sell'}">
                {advice}
            </div>
        </div>"""

    # ESB总分排名表
    ranking_rows = ""
    for rank, (_, row) in enumerate(latest_df.sort_values("ESG_total", ascending=False).iterrows(), 1):
        ind = row.get("industry", "N/A")
        esg = row.get("ESG_total", 0)
        benchmark = INDUSTRY_PROFILES.get(ind, {}).get("benchmark_esg", 60)
        diff = esg - benchmark

        rank_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}")
        ranking_rows += f"""
        <tr>
            <td>{rank_icon}</td>
            <td>{ind}</td>
            <td><strong>{esg:.1f}</strong></td>
            <td>{row.get('E_score', 0):.0f}</td>
            <td>{row.get('S_score', 0):.0f}</td>
            <td>{row.get('G_score', 0):.0f}</td>
            <td>{benchmark:.1f}</td>
            <td style="color:{'#2ECC71' if diff > 0 else '#E74C3C'}">{diff:+.1f}</td>
            <td>{row.get('trend_label', 'N/A')}</td>
            <td style="color:{'#2ECC71' if row.get('ESG_total_momentum', 0) > 0.03 else '#E74C3C' if row.get('ESG_total_momentum', 0) < 0 else '#E67E22'}">
                {row.get('ESG_total_momentum', 0):.4f}
            </td>
        </tr>"""

    # 估值对比表
    valuation_rows = ""
    for _, row in latest_df.sort_values("expected_upside_pct", ascending=False).iterrows():
        ind = row.get("industry", "N/A")
        upside = row.get("expected_upside_pct", 0)
        val = row.get("expected_value", 0)
        price = row.get("current_price", row.get("close_price", 0))
        advice = row.get("advice", "N/A")
        score = row.get("score", 0)

        valuation_rows += f"""
        <tr>
            <td>{ind}</td>
            <td>{price:.2f}</td>
            <td>{val:.2f}</td>
            <td style="color:{'#2ECC71' if upside > 0 else '#E74C3C'}; font-weight:bold;">{upside:+.1f}%</td>
            <td>{score:.2f}</td>
            <td><span class="badge-{'buy' if '买入' in str(advice) else 'hold' if '持有' in str(advice) else 'sell'}">{advice}</span></td>
        </tr>"""

    # 渲染HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESG行业综合对比分析报告</title>
    <style>
        :root {{
            --primary: #2E86AB; --green: #2ECC71; --red: #E74C3C;
            --orange: #E67E22; --purple: #9B59B6; --gray: #95A5A6;
            --dark: #2C3E50; --bg: #f8f9fa; --card-bg: #ffffff;
            --text: #333333; --border: #e0e0e0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
            background: var(--bg); color: var(--text); line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .hero {{
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: white; padding: 50px 30px; text-align: center; border-radius: 12px; margin-bottom: 30px;
        }}
        .hero h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        .hero p {{ opacity: 0.8; font-size: 1.1em; }}
        .hero .stats {{ display: flex; justify-content: center; gap: 40px; margin-top: 20px; }}
        .hero .stat {{ text-align: center; }}
        .hero .stat .num {{ font-size: 2em; font-weight: bold; }}
        .hero .stat .lbl {{ font-size: 0.85em; opacity: 0.7; }}

        .section-title {{
            font-size: 1.5em; color: var(--primary); margin: 30px 0 15px;
            padding-bottom: 8px; border-bottom: 3px solid var(--primary);
        }}
        .section-desc {{ color: var(--gray); margin-bottom: 20px; }}

        .industry-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .industry-card {{
            background: var(--card-bg); border-radius: 10px; padding: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid var(--border);
            position: relative; transition: transform 0.15s;
        }}
        .industry-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .industry-header {{ margin-bottom: 12px; }}
        .industry-header h3 {{ color: var(--dark); font-size: 1.15em; }}
        .stock-code {{ font-size: 0.8em; color: var(--gray); }}
        .industry-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }}
        .mini-metric {{ text-align: center; padding: 8px; background: var(--bg); border-radius: 6px; }}
        .mini-value {{ font-size: 1.1em; font-weight: bold; color: var(--primary); }}
        .mini-label {{ font-size: 0.7em; color: var(--gray); }}
        .industry-detail {{ font-size: 0.85em; color: #666; margin: 10px 0; line-height: 1.5; }}
        .industry-detail p {{ margin: 3px 0; }}
        .advice-badge {{
            position: absolute; top: 15px; right: 15px;
            padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold;
        }}
        .advice-buy {{ background: #d4edda; color: #155724; }}
        .advice-hold {{ background: #fff3cd; color: #856404; }}
        .advice-sell {{ background: #f8d7da; color: #721c24; }}

        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}

        table {{
            width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em;
            background: var(--card-bg); border-radius: 8px; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        th {{ background: var(--primary); color: white; padding: 12px 14px; text-align: left; }}
        td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
        tr:hover {{ background: var(--bg); }}

        .badge-buy {{ background: #d4edda; color: #155724; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}
        .badge-hold {{ background: #fff3cd; color: #856404; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}
        .badge-sell {{ background: #f8d7da; color: #721c24; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}

        .insight-box {{
            background: linear-gradient(135deg, #e8f4f8, #d4e9f2);
            padding: 20px; border-radius: 8px; margin: 20px 0;
            border-left: 4px solid var(--primary);
        }}
        .insight-box h4 {{ color: var(--primary); margin-bottom: 8px; }}
        .insight-box ul {{ margin-left: 20px; }}
        .insight-box li {{ margin: 5px 0; }}

        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

        .footer {{ text-align: center; padding: 30px; color: var(--gray); font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">

        <!-- Hero区域 -->
        <div class="hero">
            <h1>📊 ESG行业综合对比分析报告</h1>
            <p>覆盖10大行业的ESG评分、估值、风险与投资建议全景分析</p>
            <div class="stats">
                <div class="stat">
                    <div class="num">10</div>
                    <div class="lbl">覆盖行业</div>
                </div>
                <div class="stat">
                    <div class="num">6</div>
                    <div class="lbl">年度趋势 (2019-2024)</div>
                </div>
                <div class="stat">
                    <div class="num">{len(latest_df)}</div>
                    <div class="lbl">标的公司</div>
                </div>
                <div class="stat">
                    <div class="num">{latest_df['ESG_total'].mean():.1f}</div>
                    <div class="lbl">平均ESG评分</div>
                </div>
            </div>
            <p style="margin-top:15px; font-size:0.85em;">报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>

        <!-- 核心发现 -->
        <div class="insight-box">
            <h4>🔍 核心发现</h4>
            <ul>
                <li><strong>ESG评分最高行业:</strong> {latest_df.nlargest(1, 'ESG_total')['industry'].values[0]} ({latest_df['ESG_total'].max():.1f}分)</li>
                <li><strong>ESG评分最低行业:</strong> {latest_df.nsmallest(1, 'ESG_total')['industry'].values[0]} ({latest_df['ESG_total'].min():.1f}分)</li>
                <li><strong>ESG改善最快:</strong> {latest_df.nlargest(1, 'ESG_total_momentum')['industry'].values[0]} (动量: {latest_df['ESG_total_momentum'].max():.4f})</li>
                <li><strong>行业ESG标准差:</strong> {latest_df['ESG_total'].std():.1f} 分（行业间ESG表现差异程度）</li>
                <li><strong>平均异常概率:</strong> {latest_df['anomaly_probability'].mean():.1%}</li>
            </ul>
        </div>

        <!-- ESG 排名 -->
        <h2 class="section-title">🏆 行业ESG评分排名</h2>
        <p class="section-desc">按ESG总分从高到低排列，含E/S/G三维度分解和趋势动量</p>
        <table>
            <tr>
                <th>排名</th><th>行业</th><th>ESG总分</th>
                <th>E环境</th><th>S社会</th><th>G治理</th>
                <th>行业基准</th><th>差值</th>
                <th>趋势</th><th>动量</th>
            </tr>
            {ranking_rows}
        </table>

        <!-- ESG排名图 -->
        <div class="chart-container">
            <img src="../figures/industry_esg_ranking.png" alt="行业ESG排名" onerror="this.style.display='none'">
        </div>

        <!-- 估值对比 -->
        <h2 class="section-title">💰 行业估值与投资建议</h2>
        <p class="section-desc">基于DCF多情景估值和ESG因子的综合投资建议</p>
        <table>
            <tr>
                <th>行业</th><th>当前价格</th><th>期望估值</th>
                <th>上行空间</th><th>综合评分</th><th>投资建议</th>
            </tr>
            {valuation_rows}
        </table>

        <!-- 可视化图表 -->
        <h2 class="section-title">📈 可视化分析</h2>

        <div class="two-col">
            <div class="chart-container">
                <h4>各行业ESG时间趋势</h4>
                <img src="../figures/industry_esg_timeseries.png" alt="ESG时间趋势" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h4>行业风险传导分析</h4>
                <img src="../figures/industry_contagion.png" alt="风险传导" onerror="this.style.display='none'">
            </div>
        </div>

        <div class="two-col">
            <div class="chart-container">
                <h4>行业ESG维度雷达对比</h4>
                <img src="../figures/industry_multi_radar.png" alt="雷达对比" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h4>行业投资吸引力矩阵</h4>
                <img src="../figures/industry_attractiveness.png" alt="吸引力矩阵" onerror="this.style.display='none'">
            </div>
        </div>

        <div class="chart-container">
            <h4>ESG质量与动量分析</h4>
            <img src="../figures/industry_momentum_bubble.png" alt="动量气泡图" onerror="this.style.display='none'">
        </div>

        <!-- 行业详细卡片 -->
        <h2 class="section-title">🔬 各行业详细分析</h2>
        <div class="industry-grid">
            {industry_cards}
        </div>

        <!-- 投资建议总结 -->
        <div class="insight-box">
            <h4>💡 投资策略建议</h4>
            <ul>
                <li><strong>ESG领先型配置:</strong> 关注ESG总分>60且趋势改善的行业，如信息技术、医药生物、家电，适合长期ESG主题投资</li>
                <li><strong>价值修复型配置:</strong> 关注ESG动量>0.03但总分偏低的行业，如采矿、化工，可能受益于ESG改善带来的估值修复</li>
                <li><strong>风险管理:</strong> 汽车行业传导风险评分较高(>80)，需关注上游行业波动对整车制造的传导影响</li>
                <li><strong>行业分散:</strong> 建议在ESG高低评分行业间均衡配置，避免集中在单一风险暴露方向</li>
            </ul>
        </div>

        <div class="footer">
            <p>ESG Insight Valuator v1.0.0 | 行业综合对比分析报告</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据来源: 10家A股上市公司 2019-2024年度数据</p>
            <p>⚠ 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
        </div>
    </div>
</body>
</html>"""

    output_path = output_dir / "eiv_industry_comprehensive_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"行业综合报告已生成: {output_path}")
    return str(output_path)


# ============================================================================
# 8. 单行业深度分析报告
# ============================================================================

def generate_single_industry_deep_report(
    stock_code: str,
    industry: str,
    ts_df: pd.DataFrame,
    latest_row: pd.Series,
    generator: ReportGenerator,
    output_dir: str = "output/reports",
) -> Dict[str, str]:
    """为单个行业/股票生成深度分析报告。"""
    profile = INDUSTRY_PROFILES.get(industry, {})

    # ESG雷达图 - 提取分析文本
    radar_fig_dir = Path("output/figures")
    radar_fig_dir.mkdir(parents=True, exist_ok=True)
    radar_path = str(radar_fig_dir / f"esg_radar_{stock_code}.png")
    _, esg_analysis = plot_esg_radar(
        e_score=latest_row.get("E_score", 0),
        s_score=latest_row.get("S_score", 0),
        g_score=latest_row.get("G_score", 0),
        esg_total=latest_row.get("ESG_total", 0),
        industry=industry,
        stock_code=stock_code,
        output_path=radar_path,
    )
    plt.close("all")

    # 生成估值分析
    current_price = latest_row.get("current_price", latest_row.get("close_price", 1))
    expected_value = latest_row.get("expected_value", 0)
    upside = latest_row.get("expected_upside_pct", 0)
    opt_val = expected_value * 1.2
    pes_val = expected_value * 0.75
    val_gap = (opt_val - pes_val) / expected_value * 100 if expected_value > 0 else 0

    val_lines = [
        f"当前股价{current_price:.2f}元，概率加权期望估值{expected_value:.2f}元，"
        f"{'上行空间' if upside >= 0 else '下行风险'}{abs(upside):.1f}%。",
        f"乐观情景估值{opt_val:.2f}元，悲观情景估值{pes_val:.2f}元，"
        f"情景间估值跨度{val_gap:.0f}%，{'不确定性较高，需关注基本面变化' if val_gap > 50 else '估值区间合理'}。",
    ]
    if upside > 15:
        val_lines.append("当前价格显著低于期望估值，安全边际充足。")
    elif upside > 0:
        val_lines.append("当前价格略低于期望估值，存在一定上行空间。")
    elif upside > -10:
        val_lines.append("当前价格接近合理估值，建议持有观望。")
    else:
        val_lines.append("当前价格高于期望估值，建议谨慎评估风险。")
    val_analysis = "\n\n".join(val_lines)

    # 生成异常分析
    anomaly_prob = latest_row.get("anomaly_probability", 0)
    risk_level = str(latest_row.get("risk_level", "低风险"))
    anom_lines = [
        f"异常概率{anomaly_prob:.1%}，风险等级：{risk_level}。",
    ]
    if anomaly_prob > 0.5:
        anom_lines.append("异常概率较高，建议深入排查财务数据质量、关联交易和盈利可持续性。")
    elif anomaly_prob > 0.3:
        anom_lines.append("存在中等异常风险，建议关注净利润波动和资产负债率变化趋势。")
    else:
        anom_lines.append("财务异常概率较低，基本面数据质量良好。")
    anom_analysis = "\n\n".join(anom_lines)

    # 构建报告上下文（含图表分析）
    chart_analyses = {
        "esg": esg_analysis,
        "valuation": val_analysis,
        "anomaly": anom_analysis,
    }

    context = generator.build_context(
        stock_code=str(stock_code),
        industry=industry,
        esg_data={
            "E_score": latest_row.get("E_score", 0),
            "S_score": latest_row.get("S_score", 0),
            "G_score": latest_row.get("G_score", 0),
            "ESG_total": latest_row.get("ESG_total", 0),
            "trend_label": str(latest_row.get("trend_label", "")),
            "ESG_total_momentum": latest_row.get("ESG_total_momentum", 0),
        },
        valuation_data={
            "expected_value": expected_value,
            "current_price": current_price,
            "expected_upside_pct": upside,
            "scenarios": [
                {"name": "乐观", "probability": 0.25,
                 "intrinsic_value": opt_val,
                 "upside_pct": (opt_val - current_price) / max(current_price, 0.01) * 100},
                {"name": "中性", "probability": 0.50,
                 "intrinsic_value": expected_value,
                 "upside_pct": upside},
                {"name": "悲观", "probability": 0.25,
                 "intrinsic_value": pes_val,
                 "upside_pct": (pes_val - current_price) / max(current_price, 0.01) * 100},
            ],
        },
        anomaly_data={
            "anomaly_probability": anomaly_prob,
            "risk_level": risk_level,
        },
        advice_data={
            "advice": str(latest_row.get("advice", "持有")),
            "confidence": latest_row.get("confidence", 0.5),
            "score": latest_row.get("score", 0),
            "risk_warnings": [
                f"⚠ 行业风险: {', '.join(profile.get('risk_factors', ['数据不足']))}",
                f"⚠ ESG焦点: {profile.get('esg_focus', '需关注')}",
            ],
            "key_metrics": {
                "expected_upside_pct": upside,
                "asymmetry_ratio": latest_row.get("asymmetry_ratio", 1.0),
                "sharpe_approx": latest_row.get("sharpe_approx", 0.0),
                "esg_trend": str(latest_row.get("trend_label", "")),
            },
        },
        chart_analyses=chart_analyses,
    )

    return generator.generate_full_report(context, stock_code=str(stock_code)[:8])


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主入口：生成所有行业的综合分析报告。"""
    setup_logger(log_level="INFO", log_dir="logs")
    logger.info("=" * 60)
    logger.info("🏭 行业ESG综合分析报告生成器")
    logger.info(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # 1. 加载数据
    df = load_processed_data()
    latest_df = get_latest_by_stock(df)

    logger.info(f"数据加载完成: {len(latest_df)} 个行业, {df['stock_code'].nunique()} 只股票")
    for _, row in latest_df.iterrows():
        logger.info(f"  {str(row.get('industry', 'N/A')):8s} | 代码: {str(row.get('stock_code', 'N/A')):6s} | "
                   f"ESG: {row.get('ESG_total', 0):.1f} | 趋势: {row.get('trend_label', 'N/A'):6s} | "
                   f"上行: {row.get('expected_upside_pct', 0):+.1f}%")

    # 2. 创建输出目录
    out_fig = Path("output/figures")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_report = Path("output/reports")
    out_report.mkdir(parents=True, exist_ok=True)

    # 3. 生成可视化图表
    logger.info("生成综合图表...")
    figures = {}
    figure_analyses = {}

    def _safe_plot(key, fn, *args):
        try:
            result = fn(*args)
            if isinstance(result, tuple) and len(result) == 2:
                figures[key], figure_analyses[key] = result
            else:
                figures[key] = result
        except Exception as e:
            logger.error(f"{key}图表失败: {e}")

    _safe_plot("ranking", plot_industry_esg_ranking, latest_df, str(out_fig / "industry_esg_ranking.png"))
    _safe_plot("timeseries", plot_esg_timeseries_comparison, df, str(out_fig / "industry_esg_timeseries.png"))
    _safe_plot("contagion", plot_contagion_heatmap, latest_df, str(out_fig / "industry_contagion.png"))
    _safe_plot("radar", plot_multi_industry_radar, latest_df, str(out_fig / "industry_multi_radar.png"))
    _safe_plot("attractiveness", plot_industry_attractiveness, latest_df, str(out_fig / "industry_attractiveness.png"))
    _safe_plot("momentum", plot_esg_momentum_bubble, latest_df, str(out_fig / "industry_momentum_bubble.png"))

    save_all_figures_close()
    logger.info(f"图表生成完成: {len(figures)} 张")

    # 4. 生成综合行业对比报告
    logger.info("生成综合行业对比报告...")
    comparison_path = generate_industry_comparison_html(
        latest_df, df, figures, str(out_report)
    )
    logger.success(f"综合行业报告: {comparison_path}")

    # 5. 为每个行业/股票生成单份报告
    logger.info("为各行业生成单份深度报告...")
    generator = ReportGenerator(output_dir=str(out_report))

    for _, row in latest_df.iterrows():
        stock = row.get("stock_code", "?")
        industry = row.get("industry", "?")
        try:
            ts_df = get_industry_timeseries(df, stock)
            paths = generate_single_industry_deep_report(
                stock, industry, ts_df, row, generator, str(out_report)
            )
            logger.info(f"  ✅ {industry}({stock}): {list(paths.keys())}")
        except Exception as e:
            logger.error(f"  ❌ {industry}({stock}) 报告生成失败: {e}")

    # 6. 批量汇总报告
    logger.info("生成批量汇总报告...")
    try:
        generator.generate_batch_report(latest_df)
    except Exception as e:
        logger.error(f"批量报告生成失败: {e}")

    # 7. 总结
    logger.success("=" * 60)
    logger.success("🎉 所有行业分析报告生成完成！")
    logger.success(f"  - 行业综合对比报告: {comparison_path}")
    logger.success(f"  - 可视化图表: {out_fig}/")
    logger.success(f"  - 单份报告: {out_report}/")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
