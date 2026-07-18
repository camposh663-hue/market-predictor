"""Tests for the model registry."""

from __future__ import annotations

import unittest

from src.evaluation import TRAINER_FACTORIES, hyperparameter_combinations


class ModelRegistryTests(unittest.TestCase):
    def test_hyperparameter_combinations_covers_full_grid(self) -> None:
        rf_combos = hyperparameter_combinations("random_forest")
        xgb_combos = hyperparameter_combinations("xgboost")

        self.assertEqual(len(rf_combos), 4)
        self.assertEqual(len(xgb_combos), 4)
        self.assertEqual(len(rf_combos), len(set(tuple(sorted(c.items())) for c in rf_combos)))

    def test_trainer_factories_produce_the_expected_model_id(self) -> None:
        rf = TRAINER_FACTORIES["random_forest"]({"n_estimators": 5, "max_depth": 2})
        xgb = TRAINER_FACTORIES["xgboost"]({"n_estimators": 5, "max_depth": 2})

        self.assertEqual(rf.model_id, "random_forest")
        self.assertEqual(xgb.model_id, "xgboost")


if __name__ == "__main__":
    unittest.main()
