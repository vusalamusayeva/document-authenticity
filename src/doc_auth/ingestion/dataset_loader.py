"""
Dataset yükləmə və augmentation.
configs/data.yaml-dəki dataset path-larını oxuyub train/val/test üçün hazırlayır.
"""

from pathlib import Path


def load_dataset(dataset_name: str, cfg: dict) -> object:
    """cfg = configs/data.yaml-dan parse olunmuş dict."""
    raise NotImplementedError


def apply_augmentation(image, augmentation_cfg: dict) -> object:
    raise NotImplementedError


def build_splits(dataset, split_ratios: dict, stratify_on: list[str]) -> dict:
    """Bölmə 6: stratifikasiya edilmiş train/val/test bölgüsü."""
    raise NotImplementedError
