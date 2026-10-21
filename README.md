# document-authenticity

Sənəd saxtakarlığı aşkarlanması üçün çox-mərhələli (multi-stage) ML pipeline.

## Pipeline

1. **Preprocessing & Quality Check** — format, resize, denoising, keyfiyyət skoru
2. **Document Detection** — bounding box/segmentation, perspektiv düzəlişi
3. **Document Type Classification**
4. **Feature Extraction** (paralel): ELA/CNN tampering, OCR, security-element, metadata
5. **Fusion / Decision Model** — authentic / suspicious / forged + confidence
6. **Explainability Layer** — şübhəli bölgə mask-ı

Ətraflı: `configs/pipeline.yaml`

## Struktur

```
document-authenticity/
├── configs/          # data.yaml, model.yaml, pipeline.yaml
├── data/             # raw (dəyişməz) / interim / processed
├── src/doc_auth/
│   ├── ingestion/        # dataset yükləmə + document detection/cropping (addım 2)
│   ├── preprocessing/    # quality check (addım 1)
│   ├── classification/   # sənəd növü təsnifatı (addım 3)
│   ├── ocr/               # OCR + font/layout yoxlanması (addım 4b)
│   ├── forensics/         # ELA, noise-map, CNN tampering detection (addım 4a)
│   ├── signature_stamp/   # imza/möhür/security-element yoxlanması (addım 4c)
│   ├── metadata/          # EXIF/metadata analizi (addım 4d)
│   ├── fusion/            # decision model (addım 5)
│   ├── explainability/    # şübhəli bölgə mask-ı (addım 6)
│   ├── schemas/           # modullar arası daxili data strukturları
│   └── utils/             # BaseDocModel, model registry, logging
├── api/              # FastAPI backend: schemas (HTTP I/O), services, main.py
├── tests/            # unit, integration, evaluation
├── scripts/          # köməkçi skriptlər (data prep, training, s.)
├── pyproject.toml
├── Dockerfile
├── Makefile
└── .env.example
```

Qeyd: `src/doc_auth/schemas/` daxili modullar arası data strukturlarıdır (`PipelineContext`, `LabelSchema`);
`api/schemas/` isə HTTP request/response strukturlarıdır (Pydantic). İkisi qarışdırılmamalıdır.

## Quraşdırma

```bash
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env
```

## İşlətmək

```bash
make run            # API-ni lokal başlat (http://localhost:8000)
make test            # unit + integration testlər
make test-eval       # metrik-əsaslı evaluation testləri
make lint
```

Docker ilə:

```bash
make docker-build
make docker-run
```

## Coding standartları

- Python 3.11+
- Hər funksiya bir işi görür
- Bütün model class-ları `BaseDocModel` (fit/predict, sklearn-vari) interfeysinə uyğundur
  → modullar bir-birini asanlıqla əvəz edə bilər
- Minimum 70% test coverage hədəfi
- Training kodu (`scripts/`, `src/doc_auth/*`) və production inference kodu (`api/`, `src/doc_auth/inference/`) ayrıdır
- OCR preprocessing və forensic preprocessing ayrı saxlanılır (forensic analiz üçün original rəng/noise/compression izləri qorunmalıdır)

## Dataset

Bax `configs/data.yaml` və layihə sənədində Bölmə 3–6 (dataset müqayisəsi, toplama strategiyası,
labeling qaydaları, split strategiyası). Qısaca:

- Train/Val/Test: 70/15/15, stratifikasiya olunmuş (label, sənəd növü, tamper növü, mənbə)
- Label səviyyələri: document-level (authentic/forged/suspicious), region-level (mask/bbox), tamper növü, çətinlik dərəcəsi
- Hər nümunə min. 2 annotator tərəfindən işarələnir

## Model registry

`src/doc_auth/registry/model_registry.py` — hansı dataset versiyası ilə hansı model
train olunub, hansı metriklərlə nəticələnib izlənir (sadə fayl-əsaslı; production-da
MLflow/DVC ilə əvəz edilə bilər, bax `configs/model.yaml -> registry`).
