"""Registry of trainer factories and hyperparameter grids for the search.

Grids are kept intentionally small (a handful of meaningfully different
points per hyperparameter, not a dense sweep): the goal is to check whether
shallower/deeper trees or a faster/slower learning rate change the
directional-accuracy picture at all, not to squeeze out a fractional
improvement by brute force. Given how weak the signal already looked in the
first baseline, an exhaustive sweep would mostly measure noise more
precisely rather than find real gains.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from sklearn.model_selection import ParameterGrid

from src.training import ModelTrainer, RandomForestTrainer, XGBoostTrainer

TRAINER_FACTORIES: Dict[str, Callable[[Mapping[str, Any]], ModelTrainer]] = {
    "random_forest": lambda params: RandomForestTrainer(**params),
    "xgboost": lambda params: XGBoostTrainer(**params),
}

HYPERPARAMETER_GRIDS: Dict[str, Mapping[str, Sequence[Any]]] = {
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [4, 8],
    },
    "xgboost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 6],
    },
}


def hyperparameter_combinations(model_name: str) -> list:
    """Enumerate every hyperparameter combination configured for a model.

    Args:
        model_name: Key into ``HYPERPARAMETER_GRIDS``.

    Returns:
        One dict of keyword arguments per combination, suitable for passing
        to ``TRAINER_FACTORIES[model_name]``.

    Raises:
        KeyError: If ``model_name`` is not registered.
    """
    return list(ParameterGrid(dict(HYPERPARAMETER_GRIDS[model_name])))


__all__ = ["TRAINER_FACTORIES", "HYPERPARAMETER_GRIDS", "hyperparameter_combinations"]
