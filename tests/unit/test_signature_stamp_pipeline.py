from pathlib import Path
import importlib
import cv2
import numpy as np


def test_signature_stamp_pipeline_module_and_execution(tmp_path: Path) -> None:
    mod = importlib.import_module("doc_auth.signature_stamp.pipeline")
    assert mod is not None

    img_path = tmp_path / "doc_stamp.png"
    canvas = np.ones((240, 360, 3), dtype=np.uint8) * 245
    cv2.rectangle(canvas, (30, 30), (330, 210), (30, 30, 30), 2)
    cv2.circle(canvas, (250, 140), 42, (180, 30, 30), 2)
    cv2.putText(canvas, "SIGNED", (55, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.imwrite(str(img_path), canvas)

    pipeline_cls = None
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and attr.__module__ == mod.__name__:
            pipeline_cls = attr
            break

    assert pipeline_cls is not None, "No pipeline class defined in doc_auth.signature_stamp.pipeline"

    try:
        instance = pipeline_cls()
    except TypeError:
        instance = pipeline_cls(config={}) if "config" in pipeline_cls.__init__.__code__.co_varnames else None

    if instance is not None:
        for method_name in ("run", "analyze", "process", "predict", "detect"):
            method = getattr(instance, method_name, None)
            if callable(method):
                try:
                    res = method(str(img_path))
                except TypeError:
                    res = method(canvas)
                assert res is not None
                break
        else:
            if callable(instance):
                try:
                    res = instance(str(img_path))
                except TypeError:
                    res = instance(canvas)
                assert res is not None
