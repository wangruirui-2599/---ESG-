"""
自动报告生成模块
================
基于 Jinja2 模板引擎，将分析结果自动渲染为 HTML 和 Markdown 报告。

报告结构：
  1. ESG 评分概览（三维度雷达图 + 同业对比）
  2. DCF 多情景估值分析（概率加权结果）
  3. 异常检测结果（风险等级分布）
  4. 行业风险传导分析
  5. 投资建议（含置信度和风险提示）
  6. 回测绩效摘要
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from jinja2 import Environment, FileSystemLoader, select_autoescape

# ============================================================================
# 内嵌 HTML 模板（无需外部模板文件）
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESG Insight Valuator — 分析报告</title>
    <style>
        :root {
            --primary: #2E86AB;
            --green: #2ECC71;
            --red: #E74C3C;
            --orange: #E67E22;
            --purple: #9B59B6;
            --gray: #95A5A6;
            --dark: #2C3E50;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #333333;
            --border: #e0e0e0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
            background: var(--bg); color: var(--text); line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, var(--primary), var(--dark));
            color: white; padding: 40px 20px; text-align: center; border-radius: 8px; margin-bottom: 30px;
        }
        .header h1 { font-size: 2em; margin-bottom: 8px; }
        .header p { opacity: 0.85; font-size: 1.1em; }
        .card {
            background: var(--card-bg); border-radius: 8px; padding: 24px;
            margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }
        .card h2 {
            color: var(--primary); font-size: 1.4em; margin-bottom: 16px;
            padding-bottom: 8px; border-bottom: 2px solid var(--primary);
        }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
        .metric {
            text-align: center; padding: 16px; background: var(--bg);
            border-radius: 6px; border-left: 4px solid var(--primary);
        }
        .metric .value { font-size: 1.8em; font-weight: bold; color: var(--primary); }
        .metric .label { font-size: 0.85em; color: var(--gray); margin-top: 4px; }
        .metric.positive { border-left-color: var(--green); }
        .metric.positive .value { color: var(--green); }
        .metric.negative { border-left-color: var(--red); }
        .metric.negative .value { color: var(--red); }
        table {
            width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em;
        }
        th { background: var(--primary); color: white; padding: 10px 12px; text-align: left; }
        td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
        tr:hover { background: var(--bg); }
        .advice-box {
            padding: 20px; border-radius: 8px; text-align: center; margin: 16px 0;
        }
        .advice-box.buy { background: #d4edda; border: 2px solid var(--green); color: #155724; }
        .advice-box.hold { background: #fff3cd; border: 2px solid var(--orange); color: #856404; }
        .advice-box.sell { background: #f8d7da; border: 2px solid var(--red); color: #721c24; }
        .advice-box .advice-text { font-size: 2em; font-weight: bold; }
        .warnings { margin-top: 12px; padding: 12px; background: #fff3cd; border-radius: 6px; }
        .warnings li { margin: 4px 0; font-size: 0.9em; }
        .analysis-box {
            margin-top: 16px; padding: 16px;
            background: linear-gradient(135deg, #e8f4f8, #f0f7fa);
            border-radius: 6px; border-left: 4px solid var(--primary);
            font-size: 0.92em; line-height: 1.8;
        }
        .analysis-box strong { color: var(--primary); display: block; margin-bottom: 8px; }
        .analysis-box p { margin: 8px 0; }
        .footer { text-align: center; padding: 20px; color: var(--gray); font-size: 0.85em; }
        .progress-bar {
            height: 8px; border-radius: 4px; background: #e9ecef; overflow: hidden; margin: 8px 0;
        }
        .progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
    </style>
</head>
<body>
    <div class="container">

        <!-- 头部 -->
        <div class="header">
            <h1>📊 ESG Insight Valuator</h1>
            <p>ESG智能估值分析报告 | {{ report_date }}</p>
            <p style="font-size:0.85em;">标的: {{ stock_code }} | 行业: {{ industry }}</p>
        </div>

        <!-- 1. ESG评分概览 -->
        <div class="card">
            <h2>🌱 ESG 评分概览</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="value">{{ esg.E_score }}</div>
                    <div class="label">环境 (E)</div>
                </div>
                <div class="metric">
                    <div class="value">{{ esg.S_score }}</div>
                    <div class="label">社会 (S)</div>
                </div>
                <div class="metric">
                    <div class="value">{{ esg.G_score }}</div>
                    <div class="label">治理 (G)</div>
                </div>
                <div class="metric">
                    <div class="value">{{ esg.ESG_total }}</div>
                    <div class="label">ESG 综合评分</div>
                </div>
            </div>
            {% if esg.trend_label %}
            <p style="margin-top:12px;">
                📈 ESG趋势: <strong>{{ esg.trend_label }}</strong>
                (动量: {{ "%.4f"|format(esg.momentum|float) if esg.momentum else 'N/A' }})
            </p>
            {% endif %}
            {% if chart_analyses.esg %}
            <div class="analysis-box">
                <strong>📋 ESG深度分析</strong>
                <p>{{ chart_analyses.esg | replace('\n\n', '</p><p>') | safe }}</p>
            </div>
            {% endif %}
        </div>

        <!-- 2. 估值分析 -->
        <div class="card">
            <h2>💰 多情景 DCF 估值分析</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="value">{{ "%.2f"|format(valuation.expected_value|float) }}</div>
                    <div class="label">期望估值（元/股）</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.2f"|format(valuation.current_price|float) }}</div>
                    <div class="label">当前股价</div>
                </div>
                <div class="metric {{ 'positive' if valuation.expected_upside_pct > 0 else 'negative' }}">
                    <div class="value">{{ "%+.1f"|format(valuation.expected_upside_pct|float) }}%</div>
                    <div class="label">期望上行空间</div>
                </div>
            </div>

            <table>
                <tr>
                    <th>情景</th><th>概率</th><th>估值（元）</th><th>上行空间</th>
                </tr>
                {% for s in valuation.scenarios %}
                <tr>
                    <td>{{ s.name }}</td>
                    <td>{{ "%.0f"|format(s.probability*100) }}%</td>
                    <td>{{ "%.2f"|format(s.intrinsic_value|float) }}</td>
                    <td style="color:{{ '#2ECC71' if s.upside_pct > 0 else '#E74C3C' }}">
                        {{ "%+.1f"|format(s.upside_pct|float) }}%
                    </td>
                </tr>
                {% endfor %}
            </table>
            {% if chart_analyses.valuation %}
            <div class="analysis-box">
                <strong>📋 估值深度分析</strong>
                <p>{{ chart_analyses.valuation | replace('\n\n', '</p><p>') | safe }}</p>
            </div>
            {% endif %}
        </div>

        <!-- 3. 异常检测 -->
        <div class="card">
            <h2>🔍 财务异常检测</h2>
            <div class="metric-grid">
                <div class="metric {{ 'negative' if anomaly.probability > 0.3 else 'positive' }}">
                    <div class="value">{{ "%.1f"|format(anomaly.probability*100) }}%</div>
                    <div class="label">异常概率</div>
                </div>
                <div class="metric">
                    <div class="value">{{ anomaly.risk_level }}</div>
                    <div class="label">风险等级</div>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{{ anomaly.probability*100 }}%;
                    background:{{ '#2ECC71' if anomaly.probability < 0.3 else '#E67E22' if anomaly.probability < 0.6 else '#E74C3C' }};">
                </div>
            </div>
            {% if chart_analyses.anomaly %}
            <div class="analysis-box">
                <strong>📋 异常检测深度分析</strong>
                <p>{{ chart_analyses.anomaly | replace('\n\n', '</p><p>') | safe }}</p>
            </div>
            {% endif %}
        </div>

        <!-- 4. 投资建议 -->
        <div class="card">
            <h2>🎯 投资建议</h2>
            <div class="advice-box {{ advice.css_class }}">
                <div class="advice-text">{{ advice.advice_text }}</div>
                <p style="margin-top:8px;">置信度: {{ "%.1f"|format(advice.confidence*100) }}% | 综合评分: {{ "%.2f"|format(advice.score|float) }}</p>
            </div>

            {% if advice.risk_warnings %}
            <div class="warnings">
                <strong>⚠ 风险提示:</strong>
                <ul>
                {% for w in advice.risk_warnings %}
                    <li>{{ w }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}

            <div class="metric-grid" style="margin-top:16px;">
                <div class="metric">
                    <div class="value">{{ "%.2f"|format(advice.key_metrics.expected_upside_pct|float) }}%</div>
                    <div class="label">期望上行空间</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.2f"|format(advice.key_metrics.asymmetry_ratio|float) }}</div>
                    <div class="label">风险收益不对称比</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.2f"|format(advice.key_metrics.sharpe_approx|float) }}</div>
                    <div class="label">近似夏普比率</div>
                </div>
                {% if advice.key_metrics.esg_trend %}
                <div class="metric">
                    <div class="value">{{ advice.key_metrics.esg_trend }}</div>
                    <div class="label">ESG趋势</div>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- 5. 回测绩效 -->
        {% if backtest %}
        <div class="card">
            <h2>📈 策略回测</h2>
            <div class="metric-grid">
                <div class="metric {{ 'positive' if backtest.annual_return > 0 else 'negative' }}">
                    <div class="value">{{ "%.1f"|format(backtest.annual_return*100) }}%</div>
                    <div class="label">年化收益率</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.2f"|format(backtest.sharpe_ratio|float) }}</div>
                    <div class="label">夏普比率</div>
                </div>
                <div class="metric negative">
                    <div class="value">{{ "%.1f"|format(backtest.max_drawdown*100) }}%</div>
                    <div class="label">最大回撤</div>
                </div>
                <div class="metric">
                    <div class="value">{{ "%.1f"|format(backtest.win_rate*100) }}%</div>
                    <div class="label">胜率</div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- 尾部 -->
        <div class="footer">
            <p>ESG Insight Valuator v1.0.0 | 报告生成时间: {{ report_date }}</p>
            <p>本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
        </div>

    </div>
</body>
</html>"""

MARKDOWN_TEMPLATE = """# 📊 ESG Insight Valuator — 分析报告

**生成日期**: {{ report_date }}
**标的**: {{ stock_code }} | **行业**: {{ industry }}

---

## 🌱 ESG 评分概览

| 维度 | 评分 | 趋势 |
|------|------|------|
| 环境 (E) | {{ esg.E_score }} | |
| 社会 (S) | {{ esg.S_score }} | |
| 治理 (G) | {{ esg.G_score }} | |
| **ESG 综合** | **{{ esg.ESG_total }}** | {{ esg.trend_label }} |

ESG 动量: {{ "%.4f"|format(esg.momentum|float) if esg.momentum else 'N/A' }}

{% if chart_analyses.esg %}
> 📋 **ESG深度分析**
> {{ chart_analyses.esg | replace('\n\n', '\n>\n> ') }}

{% endif %}
---

## 💰 DCF 多情景估值

| 项目 | 数值 |
|------|------|
| 期望估值 | **{{ "%.2f"|format(valuation.expected_value|float) }}** 元/股 |
| 当前股价 | {{ "%.2f"|format(valuation.current_price|float) }} 元 |
| 期望上行 | {{ "%+.1f"|format(valuation.expected_upside_pct|float) }}% |

| 情景 | 概率 | 估值(元) | 上行空间 |
|------|------|----------|----------|
{% for s in valuation.scenarios %}
| {{ s.name }} | {{ "%.0f"|format(s.probability*100) }}% | {{ "%.2f"|format(s.intrinsic_value|float) }} | {{ "%+.1f"|format(s.upside_pct|float) }}% |
{% endfor %}

{% if chart_analyses.valuation %}
> 📋 **估值深度分析**
> {{ chart_analyses.valuation | replace('\n\n', '\n>\n> ') }}

{% endif %}
---

## 🔍 财务异常检测

- **异常概率**: {{ "%.1f"|format(anomaly.probability*100) }}%
- **风险等级**: {{ anomaly.risk_level }}

{% if chart_analyses.anomaly %}
> 📋 **异常检测深度分析**
> {{ chart_analyses.anomaly | replace('\n\n', '\n>\n> ') }}

{% endif %}
---

## 🎯 投资建议

> ### {{ advice.advice_text }}
>
> **置信度**: {{ "%.1f"|format(advice.confidence*100) }}% | **综合评分**: {{ "%.2f"|format(advice.score|float) }}

### 关键指标

| 指标 | 数值 |
|------|------|
| 期望上行空间 | {{ "%.2f"|format(advice.key_metrics.expected_upside_pct|float) }}% |
| 风险收益不对称比 | {{ "%.2f"|format(advice.key_metrics.asymmetry_ratio|float) }} |
| 近似夏普比率 | {{ "%.2f"|format(advice.key_metrics.sharpe_approx|float) }} |
{% if advice.key_metrics.esg_trend %}
| ESG趋势 | {{ advice.key_metrics.esg_trend }} |
{% endif %}

### ⚠ 风险提示

{% for w in advice.risk_warnings %}
- {{ w }}
{% endfor %}

---

{% if backtest %}
## 📈 策略回测绩效

| 指标 | 数值 |
|------|------|
| 年化收益率 | {{ "%.1f"|format(backtest.annual_return*100) }}% |
| 夏普比率 | {{ "%.2f"|format(backtest.sharpe_ratio|float) }} |
| 最大回撤 | {{ "%.1f"|format(backtest.max_drawdown*100) }}% |
| 胜率 | {{ "%.1f"|format(backtest.win_rate*100) }}% |
{% endif %}

---

*ESG Insight Valuator v1.0.0 | 本报告仅供参考，不构成投资建议。*
"""


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """
    自动报告生成器。

    使用 Jinja2 将分析数据渲染为 HTML / Markdown 格式报告。

    Attributes
    ----------
    output_dir : Path
        报告输出目录
    template_dir : Path
        自定义模板目录
    """

    def __init__(
        self,
        output_dir: str = "output/reports",
        template_dir: Optional[str] = None,
    ) -> None:
        """
        初始化报告生成器。

        Parameters
        ----------
        output_dir : str
            报告输出目录
        template_dir : str, optional
            自定义 Jinja2 模板目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 环境
        if template_dir and os.path.isdir(template_dir):
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self.env = Environment(autoescape=select_autoescape(["html", "xml"]))

        logger.info(f"ReportGenerator 初始化: 输出目录={self.output_dir}")

    def build_context(
        self,
        stock_code: str = "N/A",
        industry: str = "N/A",
        esg_data: Optional[Dict[str, Any]] = None,
        valuation_data: Optional[Dict[str, Any]] = None,
        anomaly_data: Optional[Dict[str, Any]] = None,
        advice_data: Optional[Dict[str, Any]] = None,
        backtest_data: Optional[Dict[str, Any]] = None,
        chart_analyses: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        构建 Jinja2 模板上下文。

        Parameters
        ----------
        stock_code : str
            股票代码
        industry : str
            行业名称
        esg_data : dict, optional
            ESG评分数据
        valuation_data : dict, optional
            估值数据
        anomaly_data : dict, optional
            异常检测数据
        advice_data : dict, optional
            投资建议数据
        backtest_data : dict, optional
            回测数据
        chart_analyses : dict, optional
            图表中文分析文本 {'esg': '...', 'valuation': '...', 'anomaly': '...'}

        Returns
        -------
        dict
            模板渲染上下文
        """
        advice = advice_data or {}
        advice_css = "hold"
        advice_text = advice.get("advice", "持有")
        if "买入" in advice_text:
            advice_css = "buy"
        elif "卖出" in advice_text:
            advice_css = "sell"

        context = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stock_code": stock_code,
            "industry": industry,
            "esg": {
                "E_score": esg_data.get("E_score", 0) if esg_data else 0,
                "S_score": esg_data.get("S_score", 0) if esg_data else 0,
                "G_score": esg_data.get("G_score", 0) if esg_data else 0,
                "ESG_total": esg_data.get("ESG_total", 0) if esg_data else 0,
                "trend_label": esg_data.get("trend_label", "") if esg_data else "",
                "momentum": esg_data.get("ESG_total_momentum", 0) if esg_data else 0,
            },
            "valuation": {
                "expected_value": valuation_data.get("expected_value", 0) if valuation_data else 0,
                "current_price": valuation_data.get("current_price", 0) if valuation_data else 0,
                "expected_upside_pct": valuation_data.get("expected_upside_pct", 0) if valuation_data else 0,
                "scenarios": valuation_data.get("scenarios", []) if valuation_data else [],
            },
            "anomaly": {
                "probability": anomaly_data.get("anomaly_probability", 0) if anomaly_data else 0,
                "risk_level": anomaly_data.get("risk_level", "未知") if anomaly_data else "未知",
            },
            "advice": {
                "advice_text": advice_text,
                "css_class": advice_css,
                "confidence": advice.get("confidence", 0),
                "score": advice.get("score", 0),
                "risk_warnings": advice.get("risk_warnings", []),
                "key_metrics": advice.get("key_metrics", {}),
            },
            "backtest": backtest_data,
            "chart_analyses": chart_analyses or {},
        }
        return context

    def generate_html(
        self,
        context: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        生成 HTML 格式报告。

        Parameters
        ----------
        context : dict
            模板上下文
        filename : str, optional
            输出文件名（不含路径）

        Returns
        -------
        str
            生成的文件路径
        """
        template = self.env.from_string(HTML_TEMPLATE)
        html_content = template.render(**context)

        if filename is None:
            stock = context.get("stock_code", "report")
            filename = f"eiv_report_{stock}_{datetime.now():%Y%m%d_%H%M%S}.html"

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML报告已生成: {output_path}")
        return str(output_path)

    def generate_markdown(
        self,
        context: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        生成 Markdown 格式报告。

        Parameters
        ----------
        context : dict
            模板上下文
        filename : str, optional
            输出文件名

        Returns
        -------
        str
            生成的文件路径
        """
        template = self.env.from_string(MARKDOWN_TEMPLATE)
        md_content = template.render(**context)

        if filename is None:
            stock = context.get("stock_code", "report")
            filename = f"eiv_report_{stock}_{datetime.now():%Y%m%d_%H%M%S}.md"

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Markdown报告已生成: {output_path}")
        return str(output_path)

    def generate_full_report(
        self,
        context: Dict[str, Any],
        stock_code: str = "report",
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        生成全格式报告（HTML + Markdown）。

        Parameters
        ----------
        context : dict
            模板上下文
        stock_code : str
            股票代码（用于文件名）
        formats : list of str, optional
            输出格式列表 ["html", "md"]，默认两种都生成

        Returns
        -------
        dict
            格式 -> 文件路径的映射
        """
        if formats is None:
            formats = ["html", "md"]

        base_name = f"eiv_report_{stock_code}_{datetime.now():%Y%m%d_%H%M%S}"
        paths = {}

        if "html" in formats:
            paths["html"] = self.generate_html(context, f"{base_name}.html")

        if "md" in formats:
            paths["md"] = self.generate_markdown(context, f"{base_name}.md")

        logger.info(f"全格式报告生成完成: {paths}")
        return paths

    def generate_batch_report(
        self,
        df_results: pd.DataFrame,
        output_filename: str = "eiv_batch_report.html",
    ) -> str:
        """
        批量生成汇总报告（所有股票在一份报告中）。

        Parameters
        ----------
        df_results : pd.DataFrame
            包含所有股票分析结果的汇总表
        output_filename : str
            输出文件名

        Returns
        -------
        str
            报告路径
        """
        n_stocks = len(df_results)

        # 构建汇总表HTML
        table_rows = ""
        for _, row in df_results.iterrows():
            advice = str(row.get("advice", "N/A"))
            upside = row.get("expected_upside_pct", 0)
            color = "#2ECC71" if upside > 0 else "#E74C3C" if upside < 0 else "#333"
            table_rows += f"""
            <tr>
                <td>{row.get('stock_code', 'N/A')}</td>
                <td>{row.get('industry', 'N/A')}</td>
                <td style="color:{color}">{upside:+.1f}%</td>
                <td>{advice}</td>
                <td>{row.get('risk_level', 'N/A')}</td>
                <td>{row.get('trend_label', 'N/A')}</td>
            </tr>"""

        batch_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>EIV 批量分析报告</title>
    <style>
        body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2E86AB; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #2E86AB; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #e0e0e0; }}
        tr:hover {{ background: #f8f9fa; }}
        .summary {{ background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>📊 ESG Insight Valuator — 批量分析报告</h1>
    <div class="summary">
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>分析股票数:</strong> {n_stocks}</p>
    </div>
    <table>
        <tr>
            <th>股票代码</th><th>行业</th><th>上行空间</th>
            <th>建议</th><th>风险等级</th><th>ESG趋势</th>
        </tr>
        {table_rows}
    </table>
    <p style="color:#95A5A6;text-align:center;margin-top:30px;">
        ESG Insight Valuator v1.0.0 | 本报告仅供参考
    </p>
</body>
</html>"""

        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(batch_html)

        logger.info(f"批量报告已生成: {output_path} ({n_stocks} 只股票)")
        return str(output_path)


# ============================================================================
# 便捷函数
# ============================================================================

def quick_report(
    stock_code: str,
    industry: str,
    esg_scores: Dict[str, float],
    dcf_value: float,
    current_price: float,
    anomaly_prob: float = 0.0,
    output_dir: str = "output/reports",
) -> str:
    """
    便捷函数：快速生成单只股票的 HTML 报告。

    Parameters
    ----------
    stock_code : str
        股票代码
    industry : str
        行业
    esg_scores : dict
        ESG评分
    dcf_value : float
        DCF估值
    current_price : float
        当前股价
    anomaly_prob : float
        异常概率
    output_dir : str
        输出目录

    Returns
    -------
    str
        报告路径
    """
    generator = ReportGenerator(output_dir)

    upside = (dcf_value - current_price) / current_price * 100 if current_price > 0 else 0

    context = generator.build_context(
        stock_code=stock_code,
        industry=industry,
        esg_data=esg_scores,
        valuation_data={
            "expected_value": dcf_value,
            "current_price": current_price,
            "expected_upside_pct": upside,
            "scenarios": [
                {"name": "乐观", "probability": 0.25, "intrinsic_value": dcf_value * 1.2,
                 "upside_pct": (dcf_value * 1.2 - current_price) / current_price * 100},
                {"name": "中性", "probability": 0.50, "intrinsic_value": dcf_value,
                 "upside_pct": upside},
                {"name": "悲观", "probability": 0.25, "intrinsic_value": dcf_value * 0.75,
                 "upside_pct": (dcf_value * 0.75 - current_price) / current_price * 100},
            ],
        },
        anomaly_data={"anomaly_probability": anomaly_prob,
                       "risk_level": "高风险" if anomaly_prob > 0.5 else "中等风险" if anomaly_prob > 0.3 else "低风险"},
        advice_data={
            "advice": "买入" if upside > 15 else "卖出" if upside < -10 else "持有",
            "confidence": 0.7, "score": upside / 5,
            "risk_warnings": ["⚠ 基于简化参数，仅供参考"],
            "key_metrics": {"expected_upside_pct": upside, "asymmetry_ratio": 2.0,
                            "sharpe_approx": 1.5, "esg_trend": "稳定"},
        },
    )

    return generator.generate_html(context)
