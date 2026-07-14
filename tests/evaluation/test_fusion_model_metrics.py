"""
Evaluation testləri — test split üzərində model metriklərini (precision/recall/F1,
confusion matrix) yoxlayır. Bu testlər CI-da yalnız release öncəsi işlədilə bilər,
çünki data/model asılıdır və vaxt apara bilər.
"""

import pytest


@pytest.mark.evaluation
def test_fusion_model_meets_minimum_recall_placeholder():
    pytest.skip("test data/processed hazır olduqdan sonra doldurun")
