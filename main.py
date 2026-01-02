# -*- coding: utf-8 -*-
"""
main.py — vfurtherProgress baseline (upload + analyze) with Hindi support.
- Keeps your proven flow:
    PDF -> identify_lab -> regex parser(s) -> Gemini extraction -> smart merge
- Adds OCR fallback: attempts to OCR scanned PDFs when structured parsing fails.
- Adds language parameter to /analyze-report/ and does JSON-preserving translation to Hindi.
- Preserves Swagger schemas for both endpoints.
"""

import os
import re
import json
import time
import fitz
import hashlib
import io
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# ---- vfurtherProgress parser imports (from your stable parserclaude.py) ----
from parser import (
    identify_lab,
    parse_lal_pathlabs,
    parse_apollo,
    parse_healthians,
    parse_awadh,
    save_debug_report,             # writes debug files
    enhanced_parse_lal_pathlabs    # alias to parse_lal_pathlabs in your file
)

# --------------------------- env & directories ------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing! Please put it in .env")
genai.configure(api_key=GOOGLE_API_KEY)

# === PASTE THE DEBUG CODE HERE ===
# --- DEBUGGING START (Print what models are actually available) ---
try:
    print(f"--- CHECKING API KEY ACCESS ({GOOGLE_API_KEY[:5]}...) ---")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    print(f"AVAILABLE MODELS: {available_models}")
except Exception as e:
    print(f"CRITICAL ERROR LISTING MODELS: {e}")
# --- DEBUGGING END ---
# =================================

DEBUG_DIR = "data/parser_debugs_v3"
os.makedirs(DEBUG_DIR, exist_ok=True)
CACHE_DIR = "data/gemini_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_THRESHOLD = 3

# ------------------------------- models -------------------------------------
class ReportDataItem(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = ""
    ref_interval: Optional[str] = ""
    source: Optional[str] = None

class ParsedReport(BaseModel):
    lab_name: str
    data: List[ReportDataItem]

class AnalyzeRequest(BaseModel):
    """
    Front-end posts:
    {
      "parsed_report": { "lab_name": "...", "data": [...] },
      "language": "en" | "hi"
    }
    """
    parsed_report: Dict[str, Any]
    language: Optional[str] = "en"

# ------------------------------ helpers -------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _cache_read(key: str):
    p = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _cache_write(key: str, obj: Any):
    p = os.path.join(CACHE_DIR, f"{key}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def _completeness(e: Dict[str, Any]) -> int:
    return sum(bool(e.get(k)) for k in ("test_name", "value", "unit", "ref_interval"))

def _n(n: str) -> str:
    return (n or "").strip().lower()

def _safe_language(lang: Optional[str]) -> str:
    return "hi" if (lang or "").lower().startswith("hi") else "en"

def _normalize_parsed(payload: Dict[str, Any]) -> ParsedReport:
    """Coerce dict from FE into ParsedReport Pydantic."""
    lab = payload.get("lab_name", "unknown") or "unknown"
    items = []
    for x in payload.get("data", []):
        items.append(ReportDataItem(
            test_name=str(x.get("test_name", "")).strip(),
            value=str(x.get("value", "")).strip(),
            unit=str(x.get("unit", "") or "").strip(),
            ref_interval=str(x.get("ref_interval", "") or "").strip(),
            source=x.get("source")
        ))
    return ParsedReport(lab_name=lab, data=items)

# --------------------------- Gemini extraction ------------------------------
async def gemini_parse(text: str) -> List[Dict[str, Any]]:
    """AI extraction (kept from your stable approach, with cache + robust JSON cleaning)."""
    if not text:
        return []

    key = _sha(text)[:32]
    cached = _cache_read(f"gp_{key}")
    if cached:
        return cached

    model = genai.GenerativeModel("gemini-flash-latest")
    prompt = f"""
You are a medical lab report parser. Extract ALL test results from the text.

Return ONLY a JSON array of objects:
[
  {{
    "test_name": "...",
    "value": "12.70",        // numeric only as string
    "unit": "g/dL",          // "" if not available
    "ref_interval": "12.00 - 15.00" // "" if not available
  }},
  ...
]

Rules:
- Include CBC, liver/kidney panels, vitamins, lipids, thyroid, etc.
- Handle concatenated formats like "g/dL12.70" -> unit="g/dL", value="12.70"
- Do NOT add commentary. Output JSON array ONLY.

TEXT:
{text[:8000]}
"""
    try:
        resp = await model.generate_content_async(prompt)
        raw = (resp.text or "").strip()

        def _try_load(s: str):
            try:
                return json.loads(s)
            except Exception:
                s2 = s.replace("```json", "").replace("```", "").strip()
                m = re.search(r'\[.*\]', s2, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
                return None

        parsed = _try_load(raw) or []
        out: List[Dict[str, Any]] = []
        for o in parsed:
            if isinstance(o, dict):
                out.append({
                    "test_name": str(o.get("test_name", "")).strip(),
                    "value": str(o.get("value", "")).strip(),
                    "unit": str(o.get("unit", "") or "").strip(),
                    "ref_interval": str(o.get("ref_interval", "") or "").strip(),
                    "source": "gemini"
                })

        _cache_write(f"gp_{key}", out)
        # keep raw for debugging
        with open(os.path.join(DEBUG_DIR, f"gemini_{key}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        return out
    except Exception as e:
        print(f"[WARN] gemini_parse failed: {e}")
        return []

# ------------------------------- merge --------------------------------------
def merge_results(gemini_results: List[Dict[str, Any]],
                  regex_results: List[Dict[str, Any]],
                  threshold: int = DEFAULT_THRESHOLD) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    # seed with AI
    for g in gemini_results:
        merged[_n(g.get("test_name", ""))] = dict(g)

    # integrate regex (replace if more complete)
    for r in regex_results:
        k = _n(r.get("test_name", ""))
        if not k:
            continue
        if k in merged:
            if _completeness(r) > _completeness(merged[k]):
                merged[k] = dict(r, source="regex")
        else:
            merged[k] = dict(r, source="regex")

    # final pass: ensure keys and stable order
    final = list(merged.values())
    final.sort(key=lambda x: (-_completeness(x), x.get("test_name", "")))

    seen = set()
    uniq = []
    for e in final:
        sig = (_n(e.get("test_name", "")), e.get("value", ""))
        if sig in seen:
            continue
        seen.add(sig)
        for k in ("test_name", "value", "unit", "ref_interval", "source"):
            e.setdefault(k, "" if k != "source" else e.get("source", "gemini"))
        uniq.append(e)
    return uniq

# ---------------------------- AI Analysis -----------------------------------
async def get_ai_analysis(parsed: ParsedReport) -> Dict[str, Any]:
    model = genai.GenerativeModel("gemini-flash-latest")

    lines = []
    for it in parsed.data:
        lines.append(f"{it.test_name} || {it.value} || {it.unit or ''} || {it.ref_interval or ''}")
    data_blob = "\n".join(lines)

    prompt = f"""
You are ReportSahayak. Given lines in the format:
test_name || value || unit || ref_interval

Return ONLY a JSON object with:
- "summary": one concise paragraph
- "details": object grouping tests into clinical buckets (e.g., "Red Blood Cells", "White Blood Cells", "Liver Function", "Kidney", "Thyroid", "Vitamins", "Lipids", etc.). Each entry is a list of items with:
   - test_name
   - value
   - status ("Low" | "Normal" | "High" | "Out of Range" | "Note")
   - analogy (patient-friendly one-liner)
   - explanation (one or two sentences, clinical but simple)
- "disclaimer": EXACT English text:
  "This is an AI-generated analysis and is for informational purposes only. It is not a substitute for professional medical advice. Please consult with a qualified doctor for any health concerns."

Use reference ranges if present to judge status, else infer cautiously.

DATA:
{data_blob}
"""
    try:
        resp = await model.generate_content_async(prompt)
        raw = (resp.text or "").strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        ai = json.loads(cleaned)

        # keep raw
        key = _sha(data_blob)[:16]
        with open(os.path.join(DEBUG_DIR, f"analysis_{key}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

        return ai
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")

# ------------------------ Hindi translation layer ---------------------------
HI_DISCLAIMER = (
    "यह एक AI-जनरेटेड विश्लेषण है और केवल जानकारी हेतु है। "
    "यह पेशेवर चिकित्सीय सलाह का विकल्प नहीं है। किसी भी स्वास्थ्य चिंता के लिए कृपया चिकित्सक से परामर्श करें।"
)
EN_DISCLAIMER = (
    "This is an AI-generated analysis and is for informational purposes only. "
    "It is not a substitute for professional medical advice. Please consult with a qualified doctor for any health concerns."
)

def _quick_hi_phrasebook(s: str) -> str:
    # very small fallback phrasebook to avoid empty output if LLM translation fails
    mapping = {
        "summary": "सारांश",
        "detailed analysis": "विस्तृत विश्लेषण",
        "red blood cells": "लाल रक्त कण",
        "white blood cells": "श्वेत रक्त कण",
        "liver function": "यकृत कार्य",
        "kidney": "गुर्दा",
        "thyroid": "थायरॉइड",
        "vitamins": "विटामिन",
        "lipids": "लिपिड"
    }
    out = s
    for en, hi in mapping.items():
        out = re.sub(rf"\b{re.escape(en)}\b", hi, out, flags=re.IGNORECASE)
    return out

async def translate_json_to_hindi(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate ONLY values (not keys) to Hindi, preserving structure and numbers.
    Uses Gemini; falls back to a tiny phrasebook if model fails.
    """
    model = genai.GenerativeModel("gemini-flash-latest")
    instruction = """
Translate the *values* of this JSON to Hindi.
- Keep the JSON structure and all KEYS exactly the same.
- Do not translate numbers, units or symbols.
- Return ONLY the translated JSON, no commentary, no code fences.
"""
    try:
        prompt = instruction + "\nJSON:\n" + json.dumps(payload, ensure_ascii=False)
        resp = await model.generate_content_async(prompt)
        raw = (resp.text or "").strip()
        s = raw.replace("```json", "").replace("```", "").strip()
        # sometimes models echo text before/after; try to extract the object
        m = re.search(r'\{.*\}\s*$', s, re.DOTALL)
        s = m.group(0) if m else s
        out = json.loads(s)
        return out
    except Exception as e:
        # fallback: shallow value translation (very conservative)
        print(f"[WARN] translate_json_to_hindi fallback: {e}")
        def walk(v):
            if isinstance(v, dict):
                return {k: walk(v[k]) for k in v}
            if isinstance(v, list):
                return [walk(x) for x in v]
            if isinstance(v, str):
                return _quick_hi_phrasebook(v)
            return v
        return walk(payload)

def add_disclaimer(result: Dict[str, Any], lang: str) -> Dict[str, Any]:
    r = dict(result)
    r["disclaimer"] = HI_DISCLAIMER if lang == "hi" else EN_DISCLAIMER
    return r

# ------------------------------- OCR integration ----------------------------
# Try to import a local OCR helper module (your ocr_implementation_patch.py)
OCR_AVAILABLE = False
ocr_processor = None
try:
    # If you provided a class named OCRProcessor in ocr_implementation_patch.py, use it.
    from ocr_implementation_patch import OCRProcessor  # <- your uploaded module (optional)
    try:
        ocr_processor = OCRProcessor()
        OCR_AVAILABLE = True
        print("[INFO] OCRProcessor loaded from ocr_implementation_patch.py")
    except Exception as e:
        print("[WARN] OCRProcessor import succeeded but initialization failed:", e)
        OCR_AVAILABLE = False
except Exception as e:
    # not fatal — we'll fall back to pytesseract if available at runtime
    print("[INFO] ocr_implementation_patch not found or failed to import:", e)
    OCR_AVAILABLE = False

async def _run_ocr_fallback(pdf_bytes: bytes, language: str = "en") -> Dict[str, Any]:
    """
    Run OCR on the PDF and attempt parsing the OCR text with the same pipeline:
      - identify_lab on OCR text
      - regex parser for detected lab
      - gemini_parse on OCR text
      - merge results
    Returns: dict { lab, data(list), ocr_text }
    Raises an Exception on failure.
    """
    lang_code = "hin" if language == "hi" else "eng"
    ocr_text = ""

    # 1) Primary: use OCRProcessor if available
    if OCR_AVAILABLE and ocr_processor is not None:
        try:
            res = ocr_processor.process_pdf_bytes(pdf_bytes, lang=lang_code)
            # support async or sync processors
            if hasattr(res, "__await__"):
                res = await res
            if isinstance(res, dict):
                # common keys: 'text', 'ocr_text', 'pages'
                ocr_text = res.get("text") or res.get("ocr_text") or "\n".join(res.get("pages", []))
            elif isinstance(res, str):
                ocr_text = res
            else:
                ocr_text = str(res or "")
            print("[INFO] OCRProcessor returned text length:", len(ocr_text or ""))
        except Exception as e:
            print("[WARN] OCRProcessor failed at runtime:", e)
            ocr_text = ""

    # 2) Fallback: direct pytesseract via PyMuPDF -> PIL images
    if not ocr_text:
        try:
            import pytesseract
            from PIL import Image
        except Exception as e:
            raise Exception(f"No OCR provider available (pytesseract not installed and OCRProcessor unavailable): {e}")

        try:
            pages_text = []
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pno, p in enumerate(doc):
                # render at reasonably high DPI for OCR
                pix = p.get_pixmap(dpi=220)
                png_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(png_bytes))
                # preprocess: convert to L (grayscale). We keep preprocessing minimal here.
                img = img.convert("L")
                # You can add thresholding/deskew here if desired.
                txt = pytesseract.image_to_string(img, lang=lang_code)
                pages_text.append(txt)
            ocr_text = "\n\n".join(pages_text)
            print("[INFO] pytesseract OCR produced length:", len(ocr_text or ""))
        except Exception as e:
            raise Exception(f"OCR (pytesseract path) failed: {e}")

    if not ocr_text or len(ocr_text.strip()) < 20:
        raise Exception("OCR produced no useful text")

    # 3) Identify lab on OCR text & run regex parsers (try to re-detect)
    lab2 = identify_lab(ocr_text)
    if lab2 == "lal_pathlabs":
        regex_results2 = enhanced_parse_lal_pathlabs(ocr_text)
    elif lab2 == "apollo":
        regex_results2 = parse_apollo(ocr_text)
    elif lab2 == "healthians":
        regex_results2 = parse_healthians(ocr_text)
    elif lab2 == "awadh":
        regex_results2 = parse_awadh(ocr_text)
    else:
        regex_results2 = []

    # 4) Gemini parse on OCR text
    ai_results2 = await gemini_parse(ocr_text)

    merged2 = merge_results(ai_results2, regex_results2, threshold=DEFAULT_THRESHOLD)

    # debug save
    try:
        save_debug_report(
            lab2,
            "uploaded_via_ocr.pdf",
            ocr_text,
            merged2,
            extra={"ocr": True, "ocr_text_len": len(ocr_text or ""), "regex_count": len(regex_results2), "gemini_count": len(ai_results2)}
        )
    except Exception as e:
        print("[WARN] save_debug_report failed for OCR fallback:", e)

    return {"lab": lab2, "data": merged2, "ocr_text": ocr_text}

# ------------------------------- FastAPI ------------------------------------
app = FastAPI(title="ReportSahayak API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <--- CHANGED: Allows ALL domains (Vercel, Localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "ReportSahayak API — vfurtherProgress + Hindi"}

# ---- 1) Upload PDF -> parse (regex + gemini merge) with OCR fallback ----------
@app.post("/upload-report/")
async def upload_report(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for p in doc:
                text += p.get_text()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF file.")

    lab = identify_lab(text)

    # Deterministic regex first (your stable parsers)
    if lab == "lal_pathlabs":
        regex_results = enhanced_parse_lal_pathlabs(text)
    elif lab == "apollo":
        regex_results = parse_apollo(text)
    elif lab == "healthians":
        regex_results = parse_healthians(text)
    elif lab == "awadh":
        regex_results = parse_awadh(text)
    else:
        regex_results = []

    # Gemini extraction
    ai_results = await gemini_parse(text)

    # Merge
    merged = merge_results(ai_results, regex_results, threshold=DEFAULT_THRESHOLD)

    # Debug log for primary parse
    try:
        save_debug_report(
            lab, getattr(file, "filename", "uploaded.pdf"), text, merged,
            extra={"regex_count": len(regex_results), "gemini_count": len(ai_results)}
        )
    except Exception as e:
        print("[WARN] save_debug_report failed for primary parse:", e)

    source_label = "hybrid"

    # If primary merge produced nothing or very few items, try OCR fallback
    try_ocr = False
    if not merged or len(merged) < 3:
        try_ocr = True

    if try_ocr:
        try:
            ocr_result = await _run_ocr_fallback(pdf_bytes, language="en")
            if ocr_result and ocr_result.get("data"):
                merged = ocr_result["data"]
                lab = ocr_result.get("lab", lab)
                source_label = "ocr_fallback"
            else:
                # if OCR returned no data, keep prior behavior: error
                raise Exception("OCR fallback parsed no fields")
        except Exception as e:
            # final failure: still return a 400 telling user OCR attempt failed
            raise HTTPException(status_code=400, detail=f"Could not parse any results from the report. OCR attempt failed: {e}")

    if not merged:
        raise HTTPException(status_code=400, detail="Could not parse any results from the report.")

    return {"lab_name": lab, "data": merged, "source": source_label}

# ---- 2) Analyze parsed JSON -> AI analysis (+ optional Hindi) --------------
@app.post("/analyze-report/")
async def analyze_report(body: Dict[str, Any]):
    """
    Accepts either:
    {
      "parsed_report": { "lab_name": "...", "data": [...] },
      "language": "en"
    }
    OR the raw parsed report directly:
    {
      "lab_name": "...",
      "data": [...]
    }
    """
    lang = _safe_language(body.get("language", "en"))

    # Unify payload
    if "parsed_report" in body:
        payload = body["parsed_report"]
    else:
        payload = body  # assume raw

    parsed = _normalize_parsed(payload)  # strict model
    analysis = await get_ai_analysis(parsed)

    # Always attach disclaimer in the target language
    analysis = add_disclaimer(analysis, lang)

    # Translate only if Hindi requested
    if lang == "hi":
        analysis = await translate_json_to_hindi(analysis)
        analysis["disclaimer"] = HI_DISCLAIMER
    return analysis

