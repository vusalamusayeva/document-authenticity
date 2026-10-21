import numpy as np
import os
import pytest
from doc_auth.signature_stamp.pipeline import analyze_signature_stamp

def test_pipeline_success():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    res = analyze_signature_stamp(img, file_id="doc_success")
    assert res.status == "success"
    assert res.file_id == "doc_success"

def test_pipeline_failed_input():
    res = analyze_signature_stamp(None, file_id="doc_fail")
    assert res.status == "failed"
    assert res.risk_score == 1.0
    assert res.is_suspicious is True

def test_pipeline_path_traversal():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res = analyze_signature_stamp(img, file_id="../../etc/passwd")
    assert res.status == "failed"
    assert res.is_suspicious is True
    assert len(res.errors) > 0

def test_pipeline_not_verified():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    res = analyze_signature_stamp(img, file_id="doc_nv", reference_signatures=None)
    assert res.signature_verifications[0]["status"] == "not_verified"

def test_pipeline_non_match():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    # Tamamilə fərqli referans imza verərək non_match statusunu təsdiqləyirik
    ref = np.zeros((300, 300, 3), dtype=np.uint8)
    res = analyze_signature_stamp(img, file_id="doc_nm", reference_signatures=[ref])
    assert res.signature_verifications[0]["status"] == "non_match"

def test_pipeline_stamp_risk():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = analyze_signature_stamp(img, file_id="doc_stamp_risk")
    # Real risk davranışını təsdiqləyirik (risk artır və ya yüksəkdir)
    assert res.risk_score > 0.0

def test_pipeline_visualization(tmp_path):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    out_dir = str(tmp_path / "outputs")
    res = analyze_signature_stamp(img, file_id="doc_vis", output_dir=out_dir)
    assert res.visualization_path is not None
    # Faylın həqiqətən yarandığını yoxlayırıq
    assert os.path.exists(res.visualization_path)

def test_pipeline_risk_threshold():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    res = analyze_signature_stamp(img, file_id="doc_thresh")
    assert 0.0 <= res.risk_score <= 1.0