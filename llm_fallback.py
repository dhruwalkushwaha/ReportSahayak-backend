# llm_fallback.py
import json, os, re, time
from typing import List, Dict, Any
import google.generativeai as genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_RETRIES = 2

SYSTEM_POLICY = (
  "Extract ONLY what is explicitly present in the text chunk. "
  "Do NOT infer, guess, or normalize units or ranges. "
  "If any field is missing, leave it as an empty string. "
  "Return strict JSON with a top-level 'items' array."
)

EXTRACTION_SCHEMA_EXAMPLE = {
  "items": [
    {
      "test_name": "Serum Bilirubin, (Total)",
      "value": "0.43",
      "unit": "mg/dl",
      "ref_interval": "0.3 - 1.2",
      "page": 2
    }
  ]
}

def _build_prompt(chunk_text: str, page_num: int, lab_hint: str) -> str:
    return f"""
{SYSTEM_POLICY}

LAB_HINT: {lab_hint}

CHUNK_PAGE: {page_num}
CHUNK_TEXT:
<<<BEGIN_CHUNK
{chunk_text}
END_CHUNK>>>

Extract lab results present in CHUNK_TEXT only.

Return ONLY JSON (no prose), following this schema exactly:
{json.dumps(EXTRACTION_SCHEMA_EXAMPLE, ensure_ascii=False)}
"""

def _clean_model_text(t: str) -> str:
    t = t.strip()
    t = t.replace("```json", "").replace("```", "").strip()
    return t

def call_gemini_chunks(chunks: List[Dict[str, Any]], lab_hint: str, api_key: str) -> List[Dict[str, str]]:
    if not api_key:
        return []
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    all_items: List[Dict[str, str]] = []
    for ch in chunks:
        prompt = _build_prompt(ch["text"], ch["page"], lab_hint)
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                resp = model.generate_content(prompt)
                raw = _clean_model_text(resp.text or "")
                data = json.loads(raw)
                if not isinstance(data, dict) or "items" not in data or not isinstance(data["items"], list):
                    raise ValueError("Bad JSON schema from model")
                # Lightweight sanity filters: require a test_name + value present in chunk
                for it in data["items"]:
                    tn = (it.get("test_name") or "").strip()
                    val = (it.get("value") or "").strip()
                    if tn and val and (tn in ch["text"]):
                        it["page"] = ch["page"]
                        all_items.append({
                            "test_name": tn,
                            "value": val,
                            "unit": (it.get("unit") or "").strip(),
                            "ref_interval": (it.get("ref_interval") or "").strip(),
                            "page": it["page"],
                            "source": "gemini_fallback"
                        })
                break
            except Exception as e:
                if attempt >= GEMINI_MAX_RETRIES:
                    # Give up on this chunk only
                    pass
                else:
                    time.sleep(0.6)
    # Deduplicate by (test_name,value,unit,ref_interval,page)
    seen = set()
    deduped = []
    for it in all_items:
        sig = (it["test_name"], it["value"], it.get("unit",""), it.get("ref_interval",""), it["page"])
        if sig not in seen:
            seen.add(sig)
            deduped.append(it)
    return deduped
