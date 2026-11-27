from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doc_auth.signature_stamp.visualization import visualize_results


class _BBoxObj:
    def __init__(self, coords):
        self._coords = coords

    def as_int_list(self):
        return list(self._coords)


def test_visualize_results_all_branches(tmp_path: Path) -> None:
    canvas = np.full((120, 160, 3), 240, dtype=np.uint8)

    candidates = [
        # Suspicious item -> red ("Subheli")
        SimpleNamespace(
            bbox=[10, 10, 60, 50],
            metadata={"is_suspicious": True},
            element_type="stamp",
            label="stamp",
            confidence=0.91,
        ),
        # Stamp item -> green ("Mohur") via as_int_list()
        SimpleNamespace(
            bbox=_BBoxObj((70, 15, 130, 55)),
            metadata={"is_suspicious": False},
            element_type="stamp",
            label=None,
            confidence=0.84,
        ),
        # Signature item -> blue ("Imza")
        SimpleNamespace(
            bbox=(20, 65, 110, 105),
            metadata={},
            element_type=None,
            label="signature",
            confidence=0.77,
        ),
        # Degenerate zero-area box -> skipped
        SimpleNamespace(
            bbox=[30, 30, 30, 40],
            metadata={},
            element_type="stamp",
            label="stamp",
            confidence=0.5,
        ),
        # Unsupported bbox object -> skipped
        SimpleNamespace(
            bbox=object(),
            metadata={},
            element_type="stamp",
            label="stamp",
            confidence=0.5,
        ),
    ]

    out_file = tmp_path / "viz_out" / "annotated.png"
    returned = visualize_results(canvas, candidates, str(out_file))  # type: ignore[arg-type]
    assert returned == str(out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    # Original input image remains untouched
    assert np.all(canvas == 240)
