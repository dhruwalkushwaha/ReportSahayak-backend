# -*- coding: utf-8 -*-
"""
main.py — ReportSahayak API (key-free edition)

Flow: PDF -> text (PyMuPDF) -> identify lab -> regex parser(s) + generic
parser -> (optional) Gemini enrichment -> local rule-based analysis ->
(optional) free translation.

Designed to work with NO API key. If GOOGLE_API_KEY is set, Gemini is used to
ENRICH extraction; if it is absent, deterministic parsers handle everything.
Analysis itself is always computed locally (see analyzer.py), so the product
never depends on an external LLM.
"""

import os
import io
import re
import json
import hashlib
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from parser import (
    identify_lab,
    parse_apollo,
    parse_healthians,
    parse_awadh,
    chaos_parser,
    smart_extract,
    enhanced_parse_lal_pathlabs,
)
import analyzer
import translation

# --------------------------- env & directories ------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Gemini is OPTIONAL. Only enabled if a key is present AND the SDK imports.
USE_LLM = False
genai = None
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GOOGLE_API_KEY)
        USE_LLM = True
        print("[INFO] GOOGLE_API_KEY detected — Gemini extraction enabled.")
    except Exception as e:
        print(f"[WARN] Gemini SDK unavailable ({e}); running in key-free mode.")
        USE_LLM = False
else:
    print("[INFO] No GOOGLE_API_KEY — running fully in key-free (local) mode.")

DEBUG_DIR = "data/parser_debugs_v3"
os.makedirs(DEBUG_DIR, exist_ok=True)
CACHE_DIR = "data/gemini_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MIN_RESULTS = 3  # below this we try harder (generic parser, then OCR)


# ------------------------------- models -------------------------------------
class ReportDataItem(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = ""
    ref_interval: Optional[str] = ""
    source: Optional[str] = None


class TranslationRequest(BaseModel):
    text: Dict[str, Any]
    target_language: str = "hi"


# ------------------------------ helpers -------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _n(s: str) -> str:
    return (s or "").strip().lower()


def _completeness(e: Dict[str, Any]) -> int:
    return sum(bool(e.get(k)) for k in ("test_name", "value", "unit", "ref_interval"))


def merge_results(primary: List[Dict[str, Any]],
                  secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge two lists of parsed items, keeping the most complete per test."""
    merged: Dict[str, Dict[str, Any]] = {}
    for g in primary:
        k = _n(g.get("test_name", ""))
        if k:
            merged[k] = dict(g)
    for r in secondary:
        k = _n(r.get("test_name", ""))
        if not k:
            continue
        if k not in merged or _completeness(r) > _completeness(merged[k]):
            merged[k] = dict(r)

    final = list(merged.values())
    final.sort(key=lambda x: (-_completeness(x), x.get("test_name", "")))
    for e in final:
        for k in ("test_name", "value", "unit", "ref_interval"):
            e.setdefault(k, "")
        e.setdefault("source", e.get("source", "regex"))
    return final


def _parse_for_lab(lab: str, text: str) -> List[Dict[str, Any]]:
    if lab == "lal_pathlabs":
        return enhanced_parse_lal_pathlabs(text)
    if lab == "apollo":
        return parse_apollo(text)
    if lab == "healthians":
        return parse_healthians(text)
    if lab == "awadh":
        return parse_awadh(text)
    return []


def _looks_real(item: Dict[str, Any]) -> bool:
    """
    Precision gate: keep an item only if it is plausibly a real lab result.
    A result must either have a parseable reference range, match a known test
    in our knowledge base, or carry a recognisable unit. This drops headers,
    addresses, names and page numbers that slip through the parsers.
    """
    name = str(item.get("test_name", "")).strip()
    if len(name) < 3 or not str(item.get("value", "")).strip():
        return False
    if analyzer.parse_ref_interval(str(item.get("ref_interval", ""))) is not None:
        return True
    if analyzer.match_test(name) is not None:
        return True
    return bool(str(item.get("unit", "")).strip())


def extract_items(text: str) -> List[Dict[str, Any]]:
    """
    Deterministic extraction with no API key: combine the lab-specific parser
    with the generic multi-line extractor, then keep only items that look like
    genuine results.
    """
    lab = identify_lab(text)
    specific = _parse_for_lab(lab, text)
    generic = smart_extract(text)
    # Trust the generic multi-line extractor first (it handles vertical layouts
    # well and carries reference ranges); use the lab-specific parser to fill
    # any gaps it leaves behind.
    combined = merge_results(generic, specific)

    filtered = [it for it in combined if _looks_real(it)]
    # If filtering nuked everything (unusual layout), fall back to the raw
    # combined set rather than returning nothing.
    return filtered or combined


# --------------------------- optional Gemini --------------------------------
async def gemini_enrich(text: str) -> List[Dict[str, Any]]:
    """Optional LLM extraction — only runs when USE_LLM is true. Never raises."""
    if not USE_LLM or not text:
        return []
    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        prompt = (
            "You are a medical lab report parser. Extract ALL test results.\n"
            "Return ONLY a JSON array of objects with keys: "
            "test_name, value, unit, ref_interval.\n\nTEXT:\n" + text[:8000]
        )
        resp = await model.generate_content_async(prompt)
        raw = (resp.text or "").strip().replace("```json", "").replace("```", "").strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else []
        out = []
        for o in parsed:
            if isinstance(o, dict) and o.get("test_name"):
                out.append({
                    "test_name": str(o.get("test_name", "")).strip(),
                    "value": str(o.get("value", "")).strip(),
                    "unit": str(o.get("unit", "") or "").strip(),
                    "ref_interval": str(o.get("ref_interval", "") or "").strip(),
                    "source": "gemini",
                })
        return out
    except Exception as e:
        print(f"[WARN] gemini_enrich failed, ignoring: {e}")
        return []


# ------------------------------- OCR fallback -------------------------------
def ocr_fallback(pdf_bytes: bytes, lang: str = "eng") -> List[Dict[str, Any]]:
    """OCR a scanned PDF and parse it. Local; requires tesseract to be present."""
    try:
        import pytesseract
        from PIL import Image
    except Exception as e:
        print(f"[INFO] OCR libraries unavailable: {e}")
        return []
    try:
        pages_text = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for p in doc:
                pix = p.get_pixmap(dpi=220)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                pages_text.append(pytesseract.image_to_string(img, lang=lang))
        ocr_text = "\n\n".join(pages_text)
    except Exception as e:
        print(f"[WARN] OCR rendering failed: {e}")
        return []
    if len(ocr_text.strip()) < 20:
        return []
    return extract_items(ocr_text)


# ------------------------------- FastAPI ------------------------------------
app = FastAPI(title="ReportSahayak API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "ReportSahayak API",
        "mode": "gemini+local" if USE_LLM else "local (no API key required)",
    }


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

    # 1) deterministic extraction
    results = extract_items(text)
    source = "local"

    # 2) optional Gemini enrichment (only if a key is configured)
    if USE_LLM:
        gem = await gemini_enrich(text)
        if gem:
            results = merge_results(gem, results)
            source = "gemini+local"

    # 3) OCR fallback for scanned/image PDFs
    if len(results) < MIN_RESULTS:
        ocr = ocr_fallback(pdf_bytes)
        if len(ocr) > len(results):
            results = ocr
            source = "ocr"

    if not results:
        raise HTTPException(
            status_code=400,
            detail="Could not read any results. Please upload a clearer, text-based PDF.",
        )

    lab = identify_lab(text)
    return {"lab_name": lab, "data": results, "source": source}


@app.post("/analyze-report/")
async def analyze_report(body: Dict[str, Any]):
    """
    Accepts: { "parsed_report": {...}, "language": "en" | "hi" | ... }
    Analysis is computed locally; translation is best-effort and never fatal.
    """
    lang = body.get("language", "en")
    payload = body.get("parsed_report", body)

    analysis = analyzer.build_analysis(payload)
    if str(lang).lower() not in ("en", "english"):
        analysis = translation.translate_analysis(analysis, lang)
    return analysis


@app.post("/translate-report/")
async def translate_report_endpoint(request: TranslationRequest):
    return translation.translate_analysis(request.text, request.target_language)
