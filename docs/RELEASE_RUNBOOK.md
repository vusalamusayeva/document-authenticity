# Release & Operations Runbook

## 1. Runtime & Endpoints
- `GET /health`: Service liveness probe (`200 OK`).
- `/deepfake/readiness`: Model availability and SHA-256 integrity check (`503` when unavailable/mismatched).
- `POST /v1/analyze`: Synchronous document authentication (`DocumentAuthSummary`).
- `POST /v1/jobs` & `GET /v1/jobs/{job_id}`: Job submission (with `idempotency_key`) and status/result lookup.

## 2. Monitoring & Privacy Rules
- Track request latency, HTTP error counts (`4xx`/`5xx`), job queue size, active model version/SHA-256, and CPU/memory usage.
- **Privacy**: Do not log raw document bytes, full extracted document text, or personal identifiers; log only `request_id`, status codes, and timings.

## 3. Rollback & Incident Response
- Roll back to the previous verified image tag and model artifact if readiness fails (`503`) or error rate spikes.
- Verify temporary upload cleanup (`tmp_uploads`) and inspect `job_queue` persisted state before restarting workers.

## 4. Known Limitations & Remaining Real-Data Calibration
- Full production calibration on a large, licensed, group-disjoint real dataset (ROC-AUC, PR-AUC, FP/FN, and fused threshold tuning) and real document training for `DocumentClassifier` should be finalized with labeled production data.
- System `tesseract` binary / `easyocr` runtime must be installed in the target runtime if live OCR text extraction (`status="success"`) is required instead of `ocr_unavailable`.
