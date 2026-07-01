from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from src.data_loading import load_dataset, paper1_dataset_specs
from src.robust_knn_graph_ensemble import RobustKNNGraphConfig, RobustKNNGraphEnsemble


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the RobustKNNGraphEnsemble method only.")
    parser.add_argument("--data-root", required=True, help="Path to the local dataset folder.")
    parser.add_argument("--dataset", choices=["all", "CICIoT23_multiclass", "EdgeIIoT_multiclass", "RT_IOT2022_attack_type"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="method_results.json", help="JSON file for proposed-method metrics.")
    args = parser.parse_args()

    selected = [s for s in paper1_dataset_specs(args.data_root) if args.dataset in ("all", s.name)]
    if not selected:
        raise ValueError(f"No dataset selected for {args.dataset}")

    results = {}
    for spec in selected:
        print(f"Loading {spec.name}")
        X_train, X_test, y_train, y_test = load_dataset(spec, seed=args.seed)
        config = RobustKNNGraphConfig(random_state=args.seed)
        model = RobustKNNGraphEnsemble(config=config)
        print(f"Training RobustKNNGraphEnsemble on {spec.name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[spec.name] = evaluate(y_test, pred)
        print(f"{spec.name}: accuracy={results[spec.name]['accuracy']:.4f}; macro_f1={results[spec.name]['macro_f1']:.4f}")

    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
