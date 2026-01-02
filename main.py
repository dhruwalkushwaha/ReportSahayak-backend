# -*- coding: utf-8 -*-
"""
main.py — vfurtherProgress baseline (upload + analyze) with Multilingual support (11 Languages).
- Keeps your proven flow: PDF -> identify_lab -> regex parser(s) -> Gemini extraction -> smart merge
- Adds OCR fallback: attempts to OCR scanned PDFs when structured parsing fails.
- Adds Universal Translation: Supports Hindi, Tamil, Telugu, Kannada, Malayalam, etc.
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
      "language": "en" | "hi" | "ta" | "te" ...
    }
    """
    parsed_report: Dict[str, Any]
    language: Optional[str] = "en"

class TranslationRequest(BaseModel):
    text: Dict[str, Any]
    target_language: str = "Hindi"

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

# ------------------------ Universal Translation ---------------------------

EN_DISCLAIMER = (
    "This is an AI-generated analysis and is for informational purposes only. "
    "It is not a substitute for professional medical advice. Please consult with a qualified doctor for any health concerns."
)

async def translate_json_payload(payload: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
    """
    Translates the values of the JSON payload to the target language.
    Keeps keys in English.
    """
    if target_lang.lower() in ["en", "english"]:
        return payload

    model = genai.GenerativeModel("gemini-flash-latest")
    
    prompt = f"""
    You are a medical translator expert in Indian languages. 
    Translate the values in the following JSON object to {target_lang}.
    
    Rules:
    1. KEEP keys in English (e.g., "Hemoglobin", "RBC Count", "summary", "details").
    2. TRANSLATE only the *values*, *descriptions*, *summaries*, and *notes* into {target_lang}.
    3. Keep medical numbers and units (e.g., "12.7 g/dL") exactly as they are.
    4. The output must be valid JSON.

    Input JSON:
    {json.dumps(payload, ensure_ascii=False)}
    """
    
    try:
        resp = await model.generate_content_async(prompt)
        raw = (resp.text or "").strip()
        cleaned_text = raw.replace("```json", "").replace("```", "").strip()
        
        # Robust parsing
        try:
            translated_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Fallback regex if extra text exists
            m = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if m:
                translated_data = json.loads(m.group(0))
            else:
                raise ValueError("Could not extract JSON from translation response")
                
        return translated_data

    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        # Return original on failure rather than crashing
        return payload

def add_disclaimer(result: Dict[str, Any], lang: str) -> Dict[str, Any]:
    r = dict(result)
    # If the translation happened, the disclaimer inside might already be translated.
    # If not, or if we want to ensure it, we can set a fallback.
    # For now, we rely on the LLM to translate the disclaimer field if present.
    if "disclaimer" not in r:
        r["disclaimer"] = EN_DISCLAIMER
    return r

# ------------------------------- OCR integration ----------------------------
OCR_AVAILABLE = False
ocr_processor = None
try:
    from ocr_implementation_patch import OCRProcessor 
    try:
        ocr_processor = OCRProcessor()
        OCR_AVAILABLE = True
        print("[INFO] OCRProcessor loaded from ocr_implementation_patch.py")
    except Exception as e:
        print("[WARN] OCRProcessor import succeeded but initialization failed:", e)
        OCR_AVAILABLE = False
except Exception as e:
    print("[INFO] ocr_implementation_patch not found or failed to import:", e)
    OCR_AVAILABLE = False

async def _run_ocr_fallback(pdf_bytes: bytes, language: str = "en") -> Dict[str, Any]:
    """
    Run OCR on the PDF and attempt parsing the OCR text.
    """
    lang_code = "hin" if language == "hi" else "eng"
    ocr_text = ""

    # 1) Primary: use OCRProcessor if available
    if OCR_AVAILABLE and ocr_processor is not None:
        try:
            res = ocr_processor.process_pdf_bytes(pdf_bytes, lang=lang_code)
            if hasattr(res, "__await__"):
                res = await res
            if isinstance(res, dict):
                ocr_text = res.get("text") or res.get("ocr_text") or "\n".join(res.get("pages", []))
            elif isinstance(res, str):
                ocr_text = res
            else:
                ocr_text = str(res or "")
        except Exception as e:
            print("[WARN] OCRProcessor failed at runtime:", e)
            ocr_text = ""

    # 2) Fallback: direct pytesseract
    if not ocr_text:
        try:
            import pytesseract
            from PIL import Image
        except Exception as e:
            raise Exception(f"No OCR provider available: {e}")

        try:
            pages_text = []
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pno, p in enumerate(doc):
                pix = p.get_pixmap(dpi=220)
                png_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(png_bytes))
                img = img.convert("L")
                txt = pytesseract.image_to_string(img, lang=lang_code)
                pages_text.append(txt)
            ocr_text = "\n\n".join(pages_text)
        except Exception as e:
            raise Exception(f"OCR (pytesseract path) failed: {e}")

    if not ocr_text or len(ocr_text.strip()) < 20:
        raise Exception("OCR produced no useful text")

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

    ai_results2 = await gemini_parse(ocr_text)
    merged2 = merge_results(ai_results2, regex_results2, threshold=DEFAULT_THRESHOLD)

    return {"lab": lab2, "data": merged2, "ocr_text": ocr_text}

# ------------------------------- FastAPI ------------------------------------
app = FastAPI(title="ReportSahayak API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "ReportSahayak API — Multilingual & OCR Enabled"}

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

    # Deterministic regex first
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

    ai_results = await gemini_parse(text)
    merged = merge_results(ai_results, regex_results, threshold=DEFAULT_THRESHOLD)

    source_label = "hybrid"
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
                raise Exception("OCR fallback parsed no fields")
        except Exception as e:
            # If no data found at all, raise error
            raise HTTPException(status_code=400, detail=f"Could not parse any results. OCR attempt failed: {e}")

    if not merged:
        raise HTTPException(status_code=400, detail="Could not parse any results from the report.")

    return {"lab_name": lab, "data": merged, "source": source_label}

# ---- 2) Analyze parsed JSON -> AI analysis (+ optional Multilingual) --------------
@app.post("/analyze-report/")
async def analyze_report(body: Dict[str, Any]):
    """
    Accepts:
    { "parsed_report": {...}, "language": "en" | "hi" | "ta" | "te" ... }
    """
    # Default to English if missing
    lang = body.get("language", "en")

    if "parsed_report" in body:
        payload = body["parsed_report"]
    else:
        payload = body 

    parsed = _normalize_parsed(payload)
    analysis = await get_ai_analysis(parsed)

    # Translate if language is NOT English
    if lang.lower() not in ["en", "english"]:
        analysis = await translate_json_payload(analysis, lang)
    
    # Ensure disclaimer exists (translated or English fallback)
    analysis = add_disclaimer(analysis, lang)
    
    return analysis

# ---- 3) Dedicated Translation Endpoint (New) ----------------------------
@app.post("/translate-report/")
async def translate_report_endpoint(request: TranslationRequest):
    """
    Directly translates a JSON analysis report to the target language.
    """
    try:
        translated = await translate_json_payload(request.text, request.target_language)
        return translated
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")