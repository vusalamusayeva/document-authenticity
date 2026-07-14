"""
Nümunə skript: processed data üzərində fusion modelini öyrədir
və nəticəni model registry-yə yazır.

İstifadə:
    python scripts/train_fusion_model.py --config configs/model.yaml
"""

import argparse

from doc_auth.fusion.decision_model import FusionDecisionModel
from doc_auth.registry.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--dataset-version", default="v0.1.0")
    args = parser.parse_args()

    # TODO: 1) data/processed-dən X, y yüklə
    #       2) FusionDecisionModel().fit(X, y)
    #       3) test split üzərində qiymətləndir
    #       4) ModelRegistry().register(...) ilə qeyd et

    registry = ModelRegistry()
    print("Skelet hazır — TODO-ları doldurun.")


if __name__ == "__main__":
    main()
