"""
LightGBM 财务异常预警模型
==========================
基于 LightGBM 梯度提升树构建企业财务异常检测模型。

功能：
  1. 自动特征筛选与训练集构建
  2. LightGBM 模型训练（带早停和交叉验证）
  3. 异常概率预测与风险分级
  4. 特征重要性分析与SHAP解释

应用场景：
  - 识别财务造假/盈余管理嫌疑企业
  - 预警潜在的ESG风险暴露导致的财务恶化
  - 为估值模型提供风险折扣因子
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 异常预测器
# ============================================================================

class AnomalyPredictor:
    """
    LightGBM 财务异常预警模型。

    Attributes
    ----------
    params : dict
        LightGBM 模型超参数
    model : lgb.Booster or None
        训练好的 LightGBM 模型
    feature_cols : list of str
        训练使用的特征列
    feature_importance_ : pd.DataFrame
        特征重要性表
    label_encoder : LabelEncoder
        标签编码器
    cv_scores_ : list of float
        交叉验证分数
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化异常预测器。

        Parameters
        ----------
        params : dict, optional
            LightGBM 超参数，默认使用内置最优参数
        """
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 7,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
            "early_stopping_rounds": 50,
        }
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: List[str] = []
        self.feature_importance_: pd.DataFrame = pd.DataFrame()
        self.label_encoder = LabelEncoder()
        self.cv_scores_: List[float] = []
        self._is_fitted = False
        logger.info(f"AnomalyPredictor 初始化: n_estimators={self.params.get('n_estimators')}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        na_strategy: str = "median",
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        准备训练/预测数据：处理缺失值、编码分类变量、筛选特征。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        label_col : str, optional
            标签列名（训练时必填，预测时可为None）
        exclude_cols : list of str, optional
            需要排除的非特征列（如股票代码、日期等）
        feature_cols : list of str, optional
            指定使用的特征列（不指定则自动选择所有数值列）
        na_strategy : str
            缺失值处理策略: "median" / "mean" / "zero" / "drop"

        Returns
        -------
        tuple
            (特征矩阵 X, 标签序列 y 或 None)
        """
        if exclude_cols is None:
            exclude_cols = ["stock_code", "report_date", "rating_date",
                           "trade_date", "trend_label", "yoy_direction"]

        df = df.copy()

        # 标签处理
        y: Optional[pd.Series] = None
        if label_col and label_col in df.columns:
            y = df[label_col].copy()
            # 编码非数值标签
            if y.dtype == object or y.dtype.name == "category":
                y = pd.Series(
                    self.label_encoder.fit_transform(y.astype(str)),
                    index=y.index,
                    name=label_col,
                )
            exclude_cols = list(set(exclude_cols + [label_col]))

        # 特征选择
        if feature_cols:
            self.feature_cols = [c for c in feature_cols if c in df.columns]
        else:
            # 自动选择数值列
            self.feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols
            ]

        # 过滤掉常量列和全空列
        valid_cols = []
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 2 or series.nunique() <= 1:
                logger.debug(f"跳过常量列/空列: {col}")
                continue
            valid_cols.append(col)

        self.feature_cols = valid_cols
        X = df[self.feature_cols].copy()

        # 处理无穷值
        X = X.replace([np.inf, -np.inf], np.nan)

        # 缺失值处理
        if na_strategy == "drop":
            mask = X.notna().all(axis=1)
            X = X[mask]
            if y is not None:
                y = y[mask]
        elif na_strategy == "zero":
            X = X.fillna(0.0)
        elif na_strategy == "mean":
            X = X.fillna(X.mean())
        elif na_strategy == "median":
            X = X.fillna(X.median())

        # 最终检查
        remaining_na = X.isna().sum().sum()
        if remaining_na > 0:
            logger.warning(f"仍有 {remaining_na} 个缺失值，用0填充")
            X = X.fillna(0.0)

        logger.info(
            f"数据准备完成: X.shape={X.shape}, "
            f"特征数={len(self.feature_cols)}, "
            f"标签分布={dict(y.value_counts()) if y is not None else 'N/A'}"
        )
        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        use_cv: bool = True,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        训练 LightGBM 异常预警模型。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签（0=正常，1=异常）
        validation_split : float
            验证集比例
        use_cv : bool
            是否使用交叉验证
        cv_folds : int
            交叉验证折数

        Returns
        -------
        dict
            训练指标: {"auc": ..., "cv_mean": ..., "cv_std": ...}
        """
        logger.info("===== 开始训练异常预警模型 =====")

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split,
            random_state=self.params.get("random_state", 42),
            stratify=y if len(y.unique()) > 1 else None,
        )

        # 构建 LightGBM Dataset
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=list(X_train.columns),
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            reference=train_data,
        )

        # 训练参数
        train_params = self.params.copy()
        n_estimators = train_params.pop("n_estimators", 200)
        early_stopping = train_params.pop("early_stopping_rounds", 50)

        # 训练
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(early_stopping),
                lgb.log_evaluation(period=50),
            ],
        )

        # 验证集评估
        y_pred_proba = self.model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred_proba)
        avg_precision = average_precision_score(y_val, y_pred_proba)

        logger.info(f"验证集 AUC={auc:.4f}, Average Precision={avg_precision:.4f}")

        # 交叉验证
        cv_mean, cv_std = 0.0, 0.0
        if use_cv:
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True,
                random_state=self.params.get("random_state", 42),
            )
            self.cv_scores_ = cross_val_score(
                lgb.LGBMClassifier(**train_params, n_estimators=n_estimators),
                X, y, cv=skf, scoring="roc_auc",
            ).tolist()
            cv_mean = np.mean(self.cv_scores_)
            cv_std = np.std(self.cv_scores_)
            logger.info(f"CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")

        # 特征重要性
        self._compute_feature_importance()

        self._is_fitted = True
        metrics = {
            "auc": round(auc, 4),
            "avg_precision": round(avg_precision, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "best_iteration": self.model.best_iteration,
        }
        logger.info(f"训练完成: {metrics}")
        return metrics

    def _compute_feature_importance(self) -> None:
        """计算并排序特征重要性。"""
        if self.model is None:
            return

        # gain 重要性
        importance_gain = self.model.feature_importance(importance_type="gain")
        importance_split = self.model.feature_importance(importance_type="split")

        self.feature_importance_ = pd.DataFrame({
            "feature": self.model.feature_name(),
            "importance_gain": importance_gain,
            "importance_split": importance_split,
        })
        # 归一化
        total_gain = self.feature_importance_["importance_gain"].sum()
        if total_gain > 0:
            self.feature_importance_["importance_gain_norm"] = (
                self.feature_importance_["importance_gain"] / total_gain
            )
        self.feature_importance_ = self.feature_importance_.sort_values(
            "importance_gain", ascending=False
        ).reset_index(drop=True)

    def predict(
        self, X: pd.DataFrame, return_proba: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        使用训练好的模型进行异常预测。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        return_proba : bool
            True 返回概率，False 返回二分类结果

        Returns
        -------
        np.ndarray or pd.DataFrame
            异常概率 (shape: n_samples,) 或包含概率和类别的 DataFrame
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        # 确保特征列匹配
        missing_cols = set(self.feature_cols) - set(X.columns)
        if missing_cols:
            logger.warning(f"缺少 {len(missing_cols)} 个特征列，以0填充")
            for col in missing_cols:
                X[col] = 0.0

        X_pred = X[self.feature_cols].fillna(0.0)

        proba = self.model.predict(X_pred)

        if not return_proba:
            return (proba >= 0.5).astype(int)

        # 风险分级
        risk_levels = pd.cut(
            proba,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["低风险", "较低风险", "中等风险", "较高风险", "高风险"],
        )

        result = pd.DataFrame({
            "anomaly_probability": np.round(proba, 4),
            "risk_level": risk_levels,
        })

        return result

    def predict_with_explanation(
        self, X: pd.DataFrame, top_features: int = 10
    ) -> pd.DataFrame:
        """
        预测异常概率并解释主要驱动因素。

        使用特征贡献度（增益 × 特征值归一化）近似 Shapley 值。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        top_features : int
            返回贡献最大的特征数

        Returns
        -------
        pd.DataFrame
            包含预测概率和 Top 驱动因素的数据表
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        proba = self.predict(X)

        # 获取Top特征的重要性排序
        top_cols = self.feature_importance_.head(top_features)["feature"].tolist()

        # 为每条记录解释风险驱动因素
        explanations = []
        for idx, row in X.iterrows():
            contributions = {}
            for col in top_cols:
                if col in row.index:
                    importance = self.feature_importance_[
                        self.feature_importance_["feature"] == col
                    ]["importance_gain_norm"].values
                    imp = importance[0] if len(importance) > 0 else 0.0
                    val = row[col]
                    contributions[col] = {
                        "value": round(val, 4) if pd.notna(val) else 0.0,
                        "contribution": round(imp * abs(val if pd.notna(val) else 0), 6),
                    }

            # 按贡献度排序
            sorted_contribs = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]["contribution"]),
                reverse=True,
            )[:top_features]

            explanations.append({
                "anomaly_probability": proba.iloc[idx]["anomaly_probability"]
                if isinstance(proba, pd.DataFrame) else proba[idx],
                "risk_level": proba.iloc[idx]["risk_level"]
                if isinstance(proba, pd.DataFrame) else "",
                "top_drivers": [
                    {"feature": f, "value": v["value"], "contribution": v["contribution"]}
                    for f, v in sorted_contribs
                ],
            })

        return pd.DataFrame(explanations)

    def save_model(self, path: str) -> None:
        """
        保存模型到文件。

        Parameters
        ----------
        path : str
            模型保存路径
        """
        if self.model is None:
            raise RuntimeError("无模型可保存")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str) -> None:
        """
        从文件加载模型。

        Parameters
        ----------
        path : str
            模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self.model = lgb.Booster(model_file=path)
        self.feature_cols = self.model.feature_name()
        self._compute_feature_importance()
        self._is_fitted = True
        logger.info(f"模型已加载: {path}, 特征数={len(self.feature_cols)}")

    def get_evaluation_report(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        生成模型评估报告。

        Parameters
        ----------
        X_test : pd.DataFrame
            测试集特征
        y_test : pd.Series
            测试集标签

        Returns
        -------
        dict
            评估指标字典
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        y_pred_proba = self.model.predict(X_test[self.feature_cols].fillna(0.0))
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_proba)
        avg_prec = average_precision_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report = {
            "auc": round(auc, 4),
            "average_precision": round(avg_prec, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "threshold_0.5_accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        }

        logger.info(f"评估报告: AUC={report['auc']}, F1={report['f1_score']}")
        return report


# ============================================================================
# 便捷函数
# ============================================================================

def train_anomaly_model(
    df: pd.DataFrame,
    label_col: str = "is_anomaly",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[AnomalyPredictor, Dict[str, Any]]:
    """
    便捷函数：快速训练异常预警模型。

    Parameters
    ----------
    df : pd.DataFrame
        训练数据
    label_col : str
        标签列名
    params : dict, optional
        模型参数

    Returns
    -------
    tuple
        (训练好的预测器, 评估指标)
    """
    predictor = AnomalyPredictor(params)
    X, y = predictor.prepare_data(df, label_col=label_col)
    if y is None:
        raise ValueError(f"标签列 '{label_col}' 不存在")
    metrics = predictor.train(X, y)
    return predictor, metrics


def detect_anomalies(
    df: pd.DataFrame,
    model_path: str,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    便捷函数：加载模型并对新数据进行异常检测。

    Parameters
    ----------
    df : pd.DataFrame
        待检测数据
    model_path : str
        模型文件路径
    threshold : float
        高风险阈值

    Returns
    -------
    pd.DataFrame
        高风险企业列表
    """
    predictor = AnomalyPredictor()
    predictor.load_model(model_path)
    X, _ = predictor.prepare_data(df)
    results = predictor.predict(X)
    high_risk = results[results["anomaly_probability"] >= threshold]
    logger.info(f"检测到 {len(high_risk)} 家高风险企业 (阈值={threshold})")
    return high_risk
