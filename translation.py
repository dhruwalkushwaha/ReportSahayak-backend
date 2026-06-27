# -*- coding: utf-8 -*-
"""
translation.py — Optional, key-free multilingual support for ReportSahayak.

Uses `deep-translator`'s free Google endpoint (no API key). Everything here is
best-effort: if the library is missing or the network call fails, we silently
return the original English text so a response is ALWAYS produced.
"""

from typing import Any, Dict, List, Optional

# Our frontend's language codes already map 1:1 to Google translate codes.
SUPPORTED = {"hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml", "pa", "or"}


def _get_translator(target: str):
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=target)
    except Exception as e:  # library missing, etc.
        print(f"[INFO] Translation unavailable ({e}); returning English.")
        return None


def _translate_many(translator, texts: List[str]) -> List[str]:
    """Translate a list of strings, best-effort, preserving order/length."""
    out: List[str] = []
    for t in texts:
        if not t or not t.strip():
            out.append(t)
            continue
        try:
            out.append(translator.translate(t))
        except Exception as e:
            print(f"[WARN] translate failed for one string: {e}")
            out.append(t)  # fall back to English
    return out


def translate_analysis(analysis: Dict[str, Any], lang: Optional[str]) -> Dict[str, Any]:
    """
    Translate the user-facing text of an analysis payload into `lang`.

    We translate: summary, disclaimer, and each item's analogy + explanation.
    We intentionally KEEP test_name, value, status and category names in English
    (medical terms are commonly read in English, and the frontend relies on the
    English status words for colour-coding).
    """
    code = (lang or "en").lower()
    if code in ("en", "english") or code not in SUPPORTED:
        return analysis

    translator = _get_translator(code)
    if translator is None:
        return analysis

    result = dict(analysis)

    # Gather every string to translate in one ordered list, then map back.
    strings: List[str] = []
    index: List[Any] = []  # bookkeeping of where each string belongs

    if isinstance(result.get("summary"), str):
        index.append(("summary", None, None))
        strings.append(result["summary"])
    if isinstance(result.get("disclaimer"), str):
        index.append(("disclaimer", None, None))
        strings.append(result["disclaimer"])

    details = result.get("details", {}) or {}
    for category, items in details.items():
        for i, item in enumerate(items):
            for field in ("analogy", "explanation"):
                if isinstance(item.get(field), str):
                    index.append(("item", category, (i, field)))
                    strings.append(item[field])

    translated = _translate_many(translator, strings)

    # Write translations back into a fresh structure.
    new_details = {cat: [dict(it) for it in items] for cat, items in details.items()}
    for (kind, category, loc), text in zip(index, translated):
        if kind == "summary":
            result["summary"] = text
        elif kind == "disclaimer":
            result["disclaimer"] = text
        elif kind == "item":
            i, field = loc
            new_details[category][i][field] = text

    result["details"] = new_details
    return result
