from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.signature_stamp.pipeline import analyze_signature_stamp


def _call_pipeline(fn, image_input, file_id: str, output_dir: str):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    kwargs = {}
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.name in ("file_id", "doc_id", "document_id"):
            kwargs[p.name] = file_id
        elif p.name == "output_dir":
            kwargs[p.name] = output_dir
        elif p.name in ("image", "img", "frame", "array") and p.default is inspect.Parameter.empty:
            kwargs[p.name] = image_input
        elif p.name in ("image_path", "path", "file_path") and p.default is inspect.Parameter.empty:
            kwargs[p.name] = str(image_input)
        elif p.default is inspect.Parameter.empty:
            kwargs[p.name] = image_input
    return fn(**kwargs)


def test_analyze_signature_stamp_pipeline(tmp_path: Path) -> None:
    img_path = tmp_path / "doc_stamp.png"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = np.ones((240, 360, 3), dtype=np.uint8) * 245
    cv2.rectangle(canvas, (30, 30), (330, 210), (30, 30, 30), 2)
    cv2.circle(canvas, (250, 140), 42, (180, 30, 30), 2)
    cv2.putText(canvas, "SIGNED", (55, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.imwrite(str(img_path), canvas)

    res_path = _call_pipeline(analyze_signature_stamp, str(img_path), "doc_path_001", str(out_dir))
    assert res_path is not None

    res_arr = _call_pipeline(analyze_signature_stamp, canvas, "doc_arr_001", str(out_dir))
    assert res_arr is not None
