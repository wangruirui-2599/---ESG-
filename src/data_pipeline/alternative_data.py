"""
另类数据融合模块
================
融合舆情情感分析、绿色专利识别、供应链关系等另类数据，
将非结构化/半结构化数据转化为可用于模型的量化特征。

模块组件：
  - SentimentAnalyzer:   基于词典的舆情情感分析
  - GreenPatentMatcher:  绿色专利 IPC 分类号匹配与评分
  - SupplyChainGraph:    供应链图构建与中心度计算
  - AlternativeDataFusion: 多源另类数据融合主控
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# 中文金融情感词典（基础版，实际部署可替换为完整词典）
# ============================================================================

POSITIVE_WORDS: Set[str] = {
    # 正面词汇
    "增长", "提升", "突破", "创新", "领先", "优势", "利好", "增长",
    "扩大", "增强", "改善", "优化", "升级", "转型", "绿色", "低碳",
    "减排", "节能", "环保", "可持续", "合规", "透明", "责任",
    "优秀", "卓越", "稳健", "强劲", "反弹", "复苏", "景气",
    "分红", "回购", "增持", "买入", "超预期", "达标", "认证",
}

NEGATIVE_WORDS: Set[str] = {
    # 负面词汇
    "下降", "下滑", "亏损", "违规", "处罚", "罚款", "事故", "污染",
    "超标", "排放", "诉讼", "纠纷", "造假", "欺诈", "退市", "破产",
    "违约", "暴雷", "减持", "减持", "跑路", "暴跌", "崩盘",
    "风险", "危机", "恶化", "恶化", "衰退", "收紧", "监管",
    "问询", "警示", "通报", "批评", "质疑", "隐患", "缺陷",
}

# 强度修饰词（程度副词）
INTENSIFIERS: Dict[str, float] = {
    "非常": 1.5, "极其": 2.0, "十分": 1.5, "特别": 1.3,
    "严重": 1.8, "大幅": 1.5, "显著": 1.3, "略微": 0.5,
    "稍微": 0.5, "略有": 0.6, "较": 0.8, "相对": 0.7,
}

# 否定词（反转情感极性）
NEGATION_WORDS: Set[str] = {"不", "没", "无", "未", "非", "否", "别", "莫"}

# ============================================================================
# 绿色专利 IPC 分类号
# ============================================================================

GREEN_IPC_CODES: Dict[str, str] = {
    # IPC分类号 -> 技术领域描述
    "B01D53": "废气处理",
    "C02F": "水处理",
    "B09B": "固体废物处理",
    "B09C": "土壤修复",
    "F03D": "风力发电",
    "H01L31": "太阳能光伏",
    "H01M8": "燃料电池",
    "H01M10": "锂电池",
    "F24S": "太阳能热利用",
    "C10L5": "生物质燃料",
    "C12P7": "生物乙醇",
    "B60L": "电动汽车",
    "G01W": "气象监测",
    "E04B1": "绿色建筑",
    "F24F11": "智能节能空调",
    "C04B7": "低碳水泥",
    "C22B7": "金属回收",
    "B29B17": "塑料回收",
    "B65D65": "可降解包装",
    "Y02": "气候变化减缓技术（CPC）",
}

# ============================================================================
# 1. 舆情情感分析器
# ============================================================================

class SentimentAnalyzer:
    """
    基于词典的中文金融舆情情感分析器。

    对新闻/公告文本进行逐句分词和情感评分，
    输出文档级情感分数（-1到1）和置信度。

    Attributes
    ----------
    positive_words : set
        正面情感词典
    negative_words : set
        负面情感词典
    intensifiers : dict
        程度副词及强度系数
    negation_words : set
        否定词集合
    """

    def __init__(
        self,
        positive_words: Optional[Set[str]] = None,
        negative_words: Optional[Set[str]] = None,
        intensifiers: Optional[Dict[str, float]] = None,
        negation_words: Optional[Set[str]] = None,
    ) -> None:
        """
        初始化情感分析器。

        Parameters
        ----------
        positive_words : set, optional
            自定义正面词典
        negative_words : set, optional
            自定义负面词典
        intensifiers : dict, optional
            自定义程度副词
        negation_words : set, optional
            自定义否定词
        """
        self.positive_words = positive_words or POSITIVE_WORDS
        self.negative_words = negative_words or NEGATIVE_WORDS
        self.intensifiers = intensifiers or INTENSIFIERS
        self.negation_words = negation_words or NEGATION_WORDS
        logger.info(
            f"SentimentAnalyzer 初始化: 正面词{len(self.positive_words)}个, "
            f"负面词{len(self.negative_words)}个"
        )

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        分析单条文本的情感。

        算法流程：
        1. 按句号/分号/换行切分为句子
        2. 逐句扫描情感词，检查否定和程度修饰
        3. 聚合所有句子得分 -> 文档得分

        Parameters
        ----------
        text : str
            待分析的文本

        Returns
        -------
        dict
            {"score": float, "positive_count": int, "negative_count": int,
             "confidence": float, "word_count": int}
        """
        if not text or not isinstance(text, str):
            return {"score": 0.0, "positive_count": 0, "negative_count": 0,
                    "confidence": 0.0, "word_count": 0}

        # 简单分词（按字符扫描，实际部署建议使用 jieba）
        sentences = re.split(r"[。；\n!！?？]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        total_score = 0.0
        pos_count = 0
        neg_count = 0

        for sentence in sentences:
            sentence_score = self._score_sentence(sentence)
            if sentence_score > 0:
                pos_count += 1
            elif sentence_score < 0:
                neg_count += 1
            total_score += sentence_score

        # 归一化到 [-1, 1]
        word_count = sum(1 for _ in re.finditer(r"[一-鿿]+", text))
        if word_count > 0:
            score = np.clip(total_score / max(word_count / 10, 1), -1.0, 1.0)
        else:
            score = 0.0

        # 置信度：情感词越多，置信度越高
        total_hits = pos_count + neg_count
        confidence = min(total_hits / 10.0, 1.0) if total_hits > 0 else 0.0

        return {
            "score": round(score, 4),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "confidence": round(confidence, 4),
            "word_count": word_count,
        }

    def _score_sentence(self, sentence: str) -> float:
        """
        对单个句子进行情感评分。

        算法：
        - 扫描句子中的每个情感词
        - 检查前方2-4个字符范围内是否有否定词（翻转极性）或程度副词（放大/缩小）
        - 正向词 +1，负向词 -1，乘以修饰系数

        Parameters
        ----------
        sentence : str
            单条句子

        Returns
        -------
        float
            句子情感分数
        """
        score = 0.0

        for i, char in enumerate(sentence):
            # 检查以当前位置开头的2-4字词
            for length in range(4, 1, -1):
                if i + length > len(sentence):
                    continue
                word = sentence[i:i + length]

                if word in self.positive_words or word in self.negative_words:
                    base = 1.0 if word in self.positive_words else -1.0

                    # 检查前方修饰词
                    modifier = self._get_modifier(sentence, i)
                    # 检查前方否定词
                    negated = self._is_negated(sentence, i)

                    if negated:
                        base *= -1.0
                    score += base * modifier
                    break  # 找到匹配后跳出长度循环

        return score

    def _get_modifier(self, sentence: str, pos: int) -> float:
        """
        获取位置 pos 前方的程度副词修饰系数。

        Parameters
        ----------
        sentence : str
            句子
        pos : int
            情感词起始位置

        Returns
        -------
        float
            修饰系数（默认1.0）
        """
        # 检查前方2-4个字符
        start = max(0, pos - 4)
        prefix = sentence[start:pos]
        for word, coeff in self.intensifiers.items():
            if prefix.endswith(word):
                return coeff
        return 1.0

    def _is_negated(self, sentence: str, pos: int) -> bool:
        """
        检查位置 pos 前方是否有否定词。

        Parameters
        ----------
        sentence : str
            句子
        pos : int
            情感词起始位置

        Returns
        -------
        bool
            是否被否定
        """
        start = max(0, pos - 3)
        prefix = sentence[start:pos]
        for neg_word in self.negation_words:
            if prefix.endswith(neg_word):
                return True
        return False

    def analyze_dataframe(
        self, df: pd.DataFrame, text_col: str = "content"
    ) -> pd.DataFrame:
        """
        批量分析 DataFrame 中的文本列。

        Parameters
        ----------
        df : pd.DataFrame
            包含文本的数据表
        text_col : str
            文本列名

        Returns
        -------
        pd.DataFrame
            添加了情感分析结果列的数据表
        """
        results = df[text_col].apply(self.analyze_text).apply(pd.Series)
        df = df.copy()
        df["sentiment_score"] = results["score"]
        df["sentiment_confidence"] = results["confidence"]
        df["positive_hits"] = results["positive_count"]
        df["negative_hits"] = results["negative_count"]

        logger.info(
            f"舆情分析完成: {len(df)} 条文本, "
            f"平均情感得分={df['sentiment_score'].mean():.3f}"
        )
        return df


# ============================================================================
# 2. 绿色专利识别器
# ============================================================================

class GreenPatentMatcher:
    """
    绿色专利 IPC 分类号匹配器。

    根据国际专利分类号（IPC）匹配绿色/低碳技术专利，
    输出专利绿色度和技术细分领域标签。

    Attributes
    ----------
    green_ipc : dict
        IPC分类号 -> 技术领域映射
    """

    def __init__(
        self, green_ipc: Optional[Dict[str, str]] = None
    ) -> None:
        """
        初始化绿色专利匹配器。

        Parameters
        ----------
        green_ipc : dict, optional
            自定义绿色IPC映射
        """
        self.green_ipc = green_ipc or GREEN_IPC_CODES
        logger.info(f"GreenPatentMatcher 初始化: {len(self.green_ipc)} 个IPC类别")

    def match(self, ipc_code: str) -> Tuple[bool, str]:
        """
        判断单个 IPC 分类号是否属于绿色专利。

        Parameters
        ----------
        ipc_code : str
            IPC 分类号（如 "B01D53/00"）

        Returns
        -------
        tuple
            (是否为绿色专利, 技术领域描述)
        """
        for code, description in self.green_ipc.items():
            # 前缀匹配：IPC 分类号以绿色代码开头即为匹配
            if ipc_code.upper().startswith(code.upper()):
                return True, description
        return False, ""

    def score_patent_portfolio(
        self, ipc_list: List[str]
    ) -> Dict[str, Any]:
        """
        对专利组合进行绿色评分。

        Parameters
        ----------
        ipc_list : list of str
            专利的 IPC 分类号列表

        Returns
        -------
        dict
            {
                "green_count": 绿色专利数量,
                "total_count": 专利总数,
                "green_ratio": 绿色占比,
                "green_score": 绿色评分 (0-100),
                "technology_fields": 涉及的技术领域列表
            }
        """
        total = len(ipc_list)
        green_count = 0
        fields: Set[str] = set()

        for ipc in ipc_list:
            is_green, field = self.match(ipc)
            if is_green:
                green_count += 1
                fields.add(field)

        green_ratio = green_count / total if total > 0 else 0.0
        # 绿色评分：占比 * 技术领域多样性加成
        diversity_bonus = min(len(fields) / 5.0, 1.0) * 20
        green_score = min(green_ratio * 100 + diversity_bonus, 100.0)

        return {
            "green_count": green_count,
            "total_count": total,
            "green_ratio": round(green_ratio, 4),
            "green_score": round(green_score, 2),
            "technology_fields": sorted(fields),
        }

    def process_dataframe(
        self, df: pd.DataFrame, ipc_col: str = "ipc_code", group_col: str = "stock_code"
    ) -> pd.DataFrame:
        """
        批量处理专利数据，按公司聚合绿色专利指标。

        Parameters
        ----------
        df : pd.DataFrame
            专利数据表，每行一条专利
        ipc_col : str
            IPC分类号列名
        group_col : str
            分组列名（股票代码）

        Returns
        -------
        pd.DataFrame
            按公司聚合的绿色专利指标
        """
        results = []
        for code, group in df.groupby(group_col):
            ipc_list = group[ipc_col].dropna().tolist()
            scores = self.score_patent_portfolio(ipc_list)
            scores[group_col] = code
            results.append(scores)

        result_df = pd.DataFrame(results)
        logger.info(
            f"专利分析完成: {len(result_df)} 家公司, "
            f"平均绿色占比={result_df['green_ratio'].mean():.2%}"
        )
        return result_df


# ============================================================================
# 3. 供应链图分析器
# ============================================================================

class SupplyChainGraph:
    """
    供应链图构建与分析器。

    基于供应商-客户关系数据构建有向图，计算网络中心度指标，
    识别供应链关键节点和风险暴露程度。

    Attributes
    ----------
    graph : dict
        邻接表表示的供应链有向图 {source: [(target, weight), ...]}
    """

    def __init__(self) -> None:
        """初始化供应链图。"""
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._in_degree: Dict[str, float] = defaultdict(float)
        self._nodes: Set[str] = set()
        logger.info("SupplyChainGraph 初始化完成")

    def add_edge(
        self, source: str, target: str, weight: float = 1.0
    ) -> None:
        """
        添加供应链边（供应商 -> 客户）。

        Parameters
        ----------
        source : str
            上游节点（供应商股票代码）
        target : str
            下游节点（客户股票代码）
        weight : float
            关系权重（交易占比，0-1）
        """
        self.graph[source].append((target, weight))
        self._in_degree[target] += weight
        self._nodes.add(source)
        self._nodes.add(target)

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        source_col: str = "supplier_code",
        target_col: str = "customer_code",
        weight_col: Optional[str] = None,
    ) -> None:
        """
        从 DataFrame 批量构建供应链图。

        Parameters
        ----------
        df : pd.DataFrame
            供应链关系数据
        source_col : str
            供应商列名
        target_col : str
            客户列名
        weight_col : str, optional
            权重列名（如采购占比）
        """
        for _, row in df.iterrows():
            weight = row[weight_col] if weight_col and weight_col in df.columns else 1.0
            self.add_edge(row[source_col], row[target_col], float(weight))

        logger.info(f"供应链图构建完成: {len(self._nodes)} 个节点, "
                     f"{sum(len(v) for v in self.graph.values())} 条边")

    def get_centrality(self) -> pd.DataFrame:
        """
        计算各节点的网络中心度指标。

        包含：
        - out_degree: 出度（供应商数量加权）
        - in_degree:  入度（客户重要性加权）
        - betweenness_approx: 近似介数中心度（基于最短路径采样）

        Returns
        -------
        pd.DataFrame
            各节点的中心度指标表
        """
        results = []
        for node in sorted(self._nodes):
            out_deg = sum(w for _, w in self.graph.get(node, []))
            in_deg = self._in_degree.get(node, 0.0)

            # 近似介数中心度（简化计算：出入度之和归一化）
            n = len(self._nodes)
            betweenness = (out_deg + in_deg) / max(2 * (n - 1), 1) if n > 1 else 0.0

            results.append({
                "stock_code": node,
                "supplier_count": len(self.graph.get(node, [])),
                "customer_importance": round(in_deg, 4),
                "betweenness": round(betweenness, 4),
                "supply_chain_exposure": round(out_deg + in_deg, 4),
            })

        result_df = pd.DataFrame(results)
        # 按供应链暴露度排序
        result_df = result_df.sort_values(
            "supply_chain_exposure", ascending=False
        ).reset_index(drop=True)

        logger.info(f"中心度计算完成: {len(result_df)} 个节点")
        return result_df

    def find_critical_path(
        self, source: str, target: str, max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        BFS 搜索从 source 到 target 的最短供应链路径。

        Parameters
        ----------
        source : str
            起始节点
        target : str
            目标节点
        max_depth : int
            最大搜索深度

        Returns
        -------
        list of str or None
            路径节点列表，不可达时返回 None
        """
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if node == target:
                return path
            for neighbor, _ in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


# ============================================================================
# 4. 另类数据融合主控
# ============================================================================

class AlternativeDataFusion:
    """
    另类数据融合主控器。

    整合舆情、专利和供应链三类另类数据，
    输出统一的企业另类特征表供模型使用。

    Attributes
    ----------
    sentiment_analyzer : SentimentAnalyzer
        舆情分析器
    patent_matcher : GreenPatentMatcher
        绿色专利匹配器
    supply_chain_graph : SupplyChainGraph
        供应链图
    """

    def __init__(
        self,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        patent_matcher: Optional[GreenPatentMatcher] = None,
        supply_chain_graph: Optional[SupplyChainGraph] = None,
    ) -> None:
        """
        初始化另类数据融合器。

        Parameters
        ----------
        sentiment_analyzer : SentimentAnalyzer, optional
        patent_matcher : GreenPatentMatcher, optional
        supply_chain_graph : SupplyChainGraph, optional
        """
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.patent_matcher = patent_matcher or GreenPatentMatcher()
        self.supply_chain_graph = supply_chain_graph or SupplyChainGraph()
        logger.info("AlternativeDataFusion 初始化完成")

    def fuse(
        self,
        sentiment_df: Optional[pd.DataFrame] = None,
        patent_df: Optional[pd.DataFrame] = None,
        supply_chain_df: Optional[pd.DataFrame] = None,
        key_col: str = "stock_code",
    ) -> pd.DataFrame:
        """
        融合多源另类数据，输出统一特征表。

        Parameters
        ----------
        sentiment_df : pd.DataFrame, optional
            舆情数据
        patent_df : pd.DataFrame, optional
            专利数据
        supply_chain_df : pd.DataFrame, optional
            供应链关系数据
        key_col : str
            关联主键列名

        Returns
        -------
        pd.DataFrame
            融合后的另类特征表
        """
        fusion_frames: List[pd.DataFrame] = []

        # 1. 舆情情感特征
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_features = self.sentiment_analyzer.analyze_dataframe(sentiment_df)
            # 按股票代码聚合
            agg_sent = sentiment_features.groupby(key_col).agg(
                avg_sentiment=("sentiment_score", "mean"),
                std_sentiment=("sentiment_score", "std"),
                positive_ratio=("sentiment_score", lambda x: (x > 0).mean()),
                news_count=("sentiment_score", "count"),
            ).reset_index()
            fusion_frames.append(agg_sent)
            logger.info(f"舆情特征融合完成: {len(agg_sent)} 只股票")

        # 2. 绿色专利特征
        if patent_df is not None and not patent_df.empty:
            patent_features = self.patent_matcher.process_dataframe(
                patent_df, group_col=key_col
            )
            fusion_frames.append(patent_features)
            logger.info(f"专利特征融合完成: {len(patent_features)} 只股票")

        # 3. 供应链特征
        if supply_chain_df is not None and not supply_chain_df.empty:
            self.supply_chain_graph.build_from_dataframe(supply_chain_df)
            supply_features = self.supply_chain_graph.get_centrality()
            fusion_frames.append(supply_features)
            logger.info(f"供应链特征融合完成: {len(supply_features)} 只股票")

        # 合并所有特征
        if not fusion_frames:
            logger.warning("无另类数据可融合")
            return pd.DataFrame()

        result = fusion_frames[0]
        for frame in fusion_frames[1:]:
            result = result.merge(frame, on=key_col, how="outer")

        # 填充缺失值
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0.0)

        logger.info(f"另类数据融合完成: {len(result)} 条记录, {len(result.columns)} 个特征")
        return result
