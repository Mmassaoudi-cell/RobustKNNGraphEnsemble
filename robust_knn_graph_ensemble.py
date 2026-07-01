from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass
class RobustKNNGraphConfig:
    random_state: int = 42
    n_neighbors: int = 10
    augmentation_sigma: float = 0.012
    rf_estimators: int = 90
    et_estimators: int = 110
    xgb_estimators: int = 90
    max_categories: int = 25


class RobustKNNGraphEnsemble(BaseEstimator, ClassifierMixin):
    """Train-only similarity-graph ensemble for multiclass IDS data.

    The method builds k-nearest-neighbor graph features from training records only.
    Each sample receives a neighbor class-probability vector and a normalized
    neighbor entropy feature. Training records are then duplicated with small
    numeric perturbations before a soft-voting RF/ExtraTrees/XGBoost ensemble is
    fitted. Test-time graph features use only neighbors from the training set.
    """

    def __init__(self, config: RobustKNNGraphConfig | None = None):
        self.config = config or RobustKNNGraphConfig()

    def fit(self, X: pd.DataFrame, y: Iterable):
        X = self._clean_frame(X)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(list(y)))
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        self.numeric_columns_ = list(X.select_dtypes(include=[np.number]).columns)
        self._fit_graph_index(X, y_enc)

        X_graph = self._append_graph_features(X, train_mode=True)
        X_aug, y_aug = self._augment_numeric(X_graph, y_enc)

        self.preprocessor_ = self._build_preprocessor(X_aug)
        self.model_ = self._build_voting_model()
        pipe = Pipeline([("preprocess", self.preprocessor_), ("model", self.model_)])
        pipe.fit(X_aug, y_aug)
        self.pipeline_ = pipe
        return self

    def predict(self, X: pd.DataFrame):
        X_graph = self._append_graph_features(self._clean_frame(X), train_mode=False)
        pred = self.pipeline_.predict(X_graph)
        return self.label_encoder_.inverse_transform(pred)

    def predict_proba(self, X: pd.DataFrame):
        X_graph = self._append_graph_features(self._clean_frame(X), train_mode=False)
        return self.pipeline_.predict_proba(X_graph)

    def _clean_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.columns = [str(c).strip() for c in X.columns]
        return X.replace([np.inf, -np.inf], np.nan)

    def _fit_graph_index(self, X: pd.DataFrame, y_enc: np.ndarray) -> None:
        self.train_labels_ = np.asarray(y_enc)
        if not self.numeric_columns_:
            self.numeric_imputer_ = None
            self.numeric_scaler_ = None
            self.train_numeric_ = np.zeros((len(X), 1), dtype=float)
        else:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            self.numeric_scaler_ = StandardScaler()
            numeric = self.numeric_imputer_.fit_transform(X[self.numeric_columns_])
            self.train_numeric_ = self.numeric_scaler_.fit_transform(numeric)
        self.graph_index_ = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors + 1, len(self.train_numeric_)),
            metric="euclidean",
        )
        self.graph_index_.fit(self.train_numeric_)

    def _transform_numeric_for_graph(self, X: pd.DataFrame) -> np.ndarray:
        if not self.numeric_columns_:
            return np.zeros((len(X), 1), dtype=float)
        numeric = self.numeric_imputer_.transform(X[self.numeric_columns_])
        return self.numeric_scaler_.transform(numeric)

    def _append_graph_features(self, X: pd.DataFrame, train_mode: bool) -> pd.DataFrame:
        numeric = self._transform_numeric_for_graph(X)
        n_neighbors = min(
            self.config.n_neighbors + 1 if train_mode else self.config.n_neighbors,
            len(self.train_numeric_),
        )
        indices = self.graph_index_.kneighbors(numeric, n_neighbors=n_neighbors, return_distance=False)
        rows = []
        for i, neigh in enumerate(indices):
            if train_mode:
                neigh = neigh[neigh != i][: self.config.n_neighbors]
            labels = self.train_labels_[neigh]
            counts = np.bincount(labels, minlength=self.n_classes_).astype(float)
            probs = counts / max(counts.sum(), 1.0)
            active = probs[probs > 0]
            entropy = -(active * np.log2(active)).sum() / max(np.log2(self.n_classes_), 1.0)
            rows.append(np.r_[probs, entropy])
        cols = [f"knn_class_probability_{i}" for i in range(self.n_classes_)] + ["knn_entropy"]
        graph = pd.DataFrame(rows, columns=cols, index=X.index)
        return pd.concat([X.reset_index(drop=True), graph.reset_index(drop=True)], axis=1)

    def _augment_numeric(self, X: pd.DataFrame, y: np.ndarray):
        rng = np.random.default_rng(self.config.random_state)
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        perturbed = X.copy()
        if numeric_cols:
            std = X[numeric_cols].astype(float).std().replace(0, 1).fillna(1).values
            noise = rng.normal(0, self.config.augmentation_sigma, size=(len(X), len(numeric_cols)))
            perturbed.loc[:, numeric_cols] = X[numeric_cols].astype(float).values + noise * std
        return pd.concat([X, perturbed], ignore_index=True), np.r_[y, y]

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric = list(X.select_dtypes(include=[np.number]).columns)
        categorical = [c for c in X.columns if c not in numeric]
        return ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=self.config.max_categories)),
                        ]
                    ),
                    categorical,
                ),
            ],
            sparse_threshold=0.3,
        )

    def _build_voting_model(self) -> VotingClassifier:
        seed = self.config.random_state
        estimators = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=self.config.rf_estimators,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=self.config.et_estimators,
                    max_depth=22,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
        weights = [1.1, 0.8]
        if XGBClassifier is not None:
            estimators.append(
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=self.config.xgb_estimators,
                        max_depth=5,
                        learning_rate=0.07,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        num_class=self.n_classes_,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=2,
                    ),
                )
            )
            weights.append(2.2)
        return VotingClassifier(estimators=estimators, voting="soft", weights=weights)
