# ReportSahayak — Backend

FastAPI service that turns a lab-report PDF into a patient-friendly,
plain-language explanation. It is built around **blood / pathology reports**
(CBC, liver, kidney, thyroid, lipids, vitamins, etc.).

## Works with no API key

The whole pipeline runs **without any external API key**:

1. **Extract** text from the PDF (PyMuPDF), with an OCR fallback for scanned
   reports (Tesseract, optional).
2. **Parse** results using lab-specific parsers plus a generic multi-line
   extractor that reconstructs `test name → value → reference range` triples
   across most lab layouts.
3. **Analyze** locally (`analyzer.py`): each value is compared against its
   printed reference range to compute **Low / Normal / High**, and a built-in
   medical knowledge base supplies the friendly analogy + explanation. No LLM
   is involved.
4. **Translate** (optional, best-effort): the analysis can be translated into
   12 Indian languages via a free, key-free translator. If the host can't reach
   the translation endpoint it silently falls back to English — a response is
   always returned.

Setting `GOOGLE_API_KEY` is **optional** and only enables extra Gemini-based
extraction enrichment; the app is fully functional without it.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 10000
# open http://localhost:10000/docs
```

Point the frontend at it with `NEXT_PUBLIC_API_URL=http://localhost:10000`.

## Run with Docker (includes Tesseract OCR + Hindi pack)

```bash
docker build -t reportsahayak-backend .
docker run -p 10000:10000 reportsahayak-backend
```

The container respects the platform-provided `$PORT` (Render / Heroku) and
falls back to `10000` locally.

## API

| Method | Path                | Purpose                                            |
| ------ | ------------------- | -------------------------------------------------- |
| GET    | `/`                 | Health check + current mode                        |
| POST   | `/upload-report/`   | multipart `file` (PDF) → parsed `{lab_name, data}` |
| POST   | `/analyze-report/`  | `{parsed_report, language}` → analysis             |
| POST   | `/translate-report/`| `{text, target_language}` → translated analysis    |
