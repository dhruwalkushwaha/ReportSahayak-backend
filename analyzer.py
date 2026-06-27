# -*- coding: utf-8 -*-
"""
analyzer.py — Local, key-free analysis engine for ReportSahayak.

This replaces the Gemini-based analysis so the app works with NO API key:
  * Status (Low / Normal / High) is computed by comparing the measured value
    against the reference interval printed on the report (pure arithmetic).
  * Patient-friendly analogies and explanations come from a built-in knowledge
    base of common lab tests, with sensible generic fallbacks for anything
    unknown.
  * The output schema is identical to what the frontend already expects:
        { "summary": str, "details": { category: [item, ...] }, "disclaimer": str }
    where each item is { test_name, value, status, analogy, explanation }.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

DISCLAIMER = (
    "This is an automated analysis and is for informational purposes only. "
    "It is not a substitute for professional medical advice. "
    "Please consult with a qualified doctor for any health concerns."
)

# --------------------------------------------------------------------------
# Reference-range parsing + status computation
# --------------------------------------------------------------------------

_NUM = r"[-+]?\d+(?:\.\d+)?"


def _first_number(s: str) -> Optional[float]:
    """Pull the first numeric token out of a string like '12.7 g/dL' -> 12.7."""
    if not s:
        return None
    m = re.search(_NUM, s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_ref_interval(ref: str) -> Optional[Dict[str, Optional[float]]]:
    """
    Parse a reference interval string into bounds.

    Returns a dict {"low": float|None, "high": float|None} or None if it
    cannot be understood. Handles the common shapes seen on Indian lab reports:
        "12.00 - 15.00"   -> low=12, high=15
        "0.3-1.2"         -> low=0.3, high=1.2
        "< 200" / "<200"  -> high=200
        "> 40"            -> low=40
        "Up to 5.0"       -> high=5.0
        "<= 5.0" / "≤ 5"  -> high=5.0
    """
    if not ref:
        return None
    s = ref.strip().replace("–", "-").replace("—", "-")
    s = s.replace("≤", "<=").replace("≥", ">=")
    low = (
        r"(?P<low>" + _NUM + r")"
    )
    high = (
        r"(?P<high>" + _NUM + r")"
    )

    # Range: "a - b"
    m = re.search(low + r"\s*-\s*" + high, s)
    if m:
        return {"low": float(m.group("low")), "high": float(m.group("high"))}

    # Upper-bound only: "< b", "<= b", "up to b"
    m = re.search(r"(?:<=?|up\s*to|upto|less than|below)\s*(?P<high>" + _NUM + r")", s, re.IGNORECASE)
    if m:
        return {"low": None, "high": float(m.group("high"))}

    # Lower-bound only: "> a", ">= a", "above a"
    m = re.search(r"(?:>=?|above|greater than|min(?:imum)?)\s*(?P<low>" + _NUM + r")", s, re.IGNORECASE)
    if m:
        return {"low": float(m.group("low")), "high": None}

    return None


def compute_status(value: str, ref_interval: str) -> str:
    """
    Compare a value against its reference interval.
    Returns one of: "Low", "Normal", "High", "Note".
    "Note" means we couldn't judge it (missing value or unparseable range).
    """
    val = _first_number(value)
    bounds = parse_ref_interval(ref_interval)
    if val is None or bounds is None:
        return "Note"

    low, high = bounds.get("low"), bounds.get("high")
    if low is not None and val < low:
        return "Low"
    if high is not None and val > high:
        return "High"
    return "Normal"


# --------------------------------------------------------------------------
# Knowledge base: canonical test -> (category, analogy, explanation)
# --------------------------------------------------------------------------
# `keywords` are matched (case-insensitively) against the report's test name.
# The first entry whose keywords are all/any present wins.

KB: List[Dict[str, Any]] = [
    # ---- Red blood cells ----
    {"keywords": ["hemoglobin", "haemoglobin", "hgb", "hb"], "category": "Red Blood Cells",
     "analogy": "Think of hemoglobin as the delivery trucks carrying oxygen around your body.",
     "explanation": "Hemoglobin is the protein in red blood cells that carries oxygen. Low levels can mean anaemia; high levels can relate to dehydration or other conditions."},
    {"keywords": ["rbc", "red blood cell", "red cell count", "erythrocyte"], "category": "Red Blood Cells",
     "analogy": "Your fleet size of oxygen-carrying trucks.",
     "explanation": "The red blood cell count reflects how many oxygen-carrying cells you have. It is read together with hemoglobin and hematocrit."},
    {"keywords": ["pcv", "packed cell volume", "hematocrit", "haematocrit"], "category": "Red Blood Cells",
     "analogy": "How much of your blood is 'solid' red cells versus liquid.",
     "explanation": "Hematocrit/PCV is the proportion of blood made up of red cells. It tracks closely with hemoglobin."},
    {"keywords": ["mcv"], "category": "Red Blood Cells",
     "analogy": "The average size of each oxygen truck.",
     "explanation": "MCV is the average size of your red blood cells and helps classify the type of anaemia."},
    {"keywords": ["mch", "mchc"], "category": "Red Blood Cells",
     "analogy": "How much oxygen cargo each truck is carrying on average.",
     "explanation": "MCH/MCHC describe the average amount and concentration of hemoglobin inside each red cell."},
    {"keywords": ["rdw"], "category": "Red Blood Cells",
     "analogy": "How much your truck sizes vary from one another.",
     "explanation": "RDW measures variation in red cell size. A high value can be an early clue to certain anaemias."},

    # ---- White blood cells ----
    {"keywords": ["tlc", "total leukocyte", "wbc", "white blood cell", "leukocyte count"], "category": "White Blood Cells",
     "analogy": "Your body's standing army of defenders.",
     "explanation": "White blood cells fight infection. High counts often suggest infection or inflammation; low counts can mean a weakened defence."},
    {"keywords": ["neutrophil"], "category": "White Blood Cells",
     "analogy": "The front-line soldiers, first to respond to bacteria.",
     "explanation": "Neutrophils are the most common white cells and rise quickly during bacterial infections."},
    {"keywords": ["lymphocyte"], "category": "White Blood Cells",
     "analogy": "The intelligence unit that remembers past invaders.",
     "explanation": "Lymphocytes handle viral defence and immune memory. They often rise during viral infections."},
    {"keywords": ["monocyte"], "category": "White Blood Cells",
     "analogy": "The clean-up crew.",
     "explanation": "Monocytes clear debris and dead cells and support longer-term immune responses."},
    {"keywords": ["eosinophil"], "category": "White Blood Cells",
     "analogy": "The allergy and parasite squad.",
     "explanation": "Eosinophils rise with allergies, asthma, or parasitic infections."},
    {"keywords": ["basophil"], "category": "White Blood Cells",
     "analogy": "A small but specialised alarm crew.",
     "explanation": "Basophils are involved in allergic and inflammatory reactions."},

    # ---- Platelets ----
    {"keywords": ["platelet", "plt"], "category": "Platelets",
     "analogy": "The repair patches that plug leaks when you bleed.",
     "explanation": "Platelets help blood clot. Very low counts raise bleeding risk; very high counts can affect clotting."},

    # ---- Diabetes / sugar ----
    {"keywords": ["hba1c", "glycated", "glycosylated"], "category": "Blood Sugar",
     "analogy": "Your average sugar 'report card' for the last 3 months.",
     "explanation": "HbA1c reflects average blood sugar over ~3 months and is key for diagnosing and monitoring diabetes."},
    {"keywords": ["glucose", "sugar", "fbs", "ppbs", "rbs"], "category": "Blood Sugar",
     "analogy": "The fuel level in your bloodstream right now.",
     "explanation": "Blood glucose shows your current sugar level. Persistently high values can indicate diabetes."},

    # ---- Lipids ----
    {"keywords": ["total cholesterol", "cholesterol total", "cholesterol, total"], "category": "Lipid Profile",
     "analogy": "The overall amount of fatty material in your blood.",
     "explanation": "Total cholesterol sums the good and bad fats. It is interpreted alongside HDL and LDL."},
    {"keywords": ["hdl"], "category": "Lipid Profile",
     "analogy": "The 'good' cleaners that carry fat away from arteries.",
     "explanation": "HDL is protective cholesterol. Higher values are generally better for heart health."},
    {"keywords": ["ldl"], "category": "Lipid Profile",
     "analogy": "The 'bad' fat that can clog pipes (arteries).",
     "explanation": "LDL can build up in arteries. Lower values are generally better for heart health."},
    {"keywords": ["vldl"], "category": "Lipid Profile",
     "analogy": "A carrier mostly hauling triglyceride fat.",
     "explanation": "VLDL transports triglycerides and is part of the bad-cholesterol picture."},
    {"keywords": ["triglyceride"], "category": "Lipid Profile",
     "analogy": "Stored energy fat floating in your blood.",
     "explanation": "Triglycerides are a type of fat; high levels are linked to heart and metabolic risk."},

    # ---- Liver ----
    {"keywords": ["sgpt", "alt", "alanine"], "category": "Liver Function",
     "analogy": "A leak-detector for liver cells.",
     "explanation": "ALT (SGPT) rises when liver cells are stressed or damaged."},
    {"keywords": ["sgot", "ast", "aspartate"], "category": "Liver Function",
     "analogy": "Another liver/muscle stress signal.",
     "explanation": "AST (SGOT) can rise with liver or muscle injury and is read with ALT."},
    {"keywords": ["bilirubin"], "category": "Liver Function",
     "analogy": "A yellow pigment your liver normally clears away.",
     "explanation": "Bilirubin is processed by the liver. High levels can cause jaundice."},
    {"keywords": ["alkaline phosphatase", "alp"], "category": "Liver Function",
     "analogy": "An enzyme from liver and bone.",
     "explanation": "ALP can rise with certain liver, bile-duct, or bone conditions."},
    {"keywords": ["albumin"], "category": "Liver Function",
     "analogy": "The main protein keeping fluid inside your vessels.",
     "explanation": "Albumin is made by the liver and reflects nutrition and liver/kidney health."},
    {"keywords": ["total protein", "protein total"], "category": "Liver Function",
     "analogy": "The total building-block proteins in your blood.",
     "explanation": "Total protein combines albumin and globulins and gives a broad nutritional/liver picture."},

    # ---- Kidney ----
    {"keywords": ["creatinine"], "category": "Kidney Function",
     "analogy": "A waste product your kidneys should be filtering out.",
     "explanation": "Creatinine reflects kidney filtering. Higher values can indicate reduced kidney function."},
    {"keywords": ["urea", "bun", "blood urea"], "category": "Kidney Function",
     "analogy": "Another waste product cleared by the kidneys.",
     "explanation": "Urea/BUN rises when kidneys filter less effectively or with dehydration."},
    {"keywords": ["uric acid"], "category": "Kidney Function",
     "analogy": "A waste that can crystallise and cause gout.",
     "explanation": "High uric acid is linked to gout and kidney stones."},

    # ---- Thyroid ----
    {"keywords": ["tsh"], "category": "Thyroid",
     "analogy": "The thermostat signal telling your thyroid to work harder or ease off.",
     "explanation": "TSH is the master control for the thyroid. High TSH often means an underactive thyroid; low TSH the opposite."},
    {"keywords": ["t3", "triiodothyronine"], "category": "Thyroid",
     "analogy": "An active thyroid hormone setting your metabolism.",
     "explanation": "T3 is an active thyroid hormone that regulates metabolism."},
    {"keywords": ["t4", "thyroxine"], "category": "Thyroid",
     "analogy": "The main thyroid hormone in circulation.",
     "explanation": "T4 is the main thyroid hormone, converted to active T3 in the body."},

    # ---- Vitamins / minerals ----
    {"keywords": ["vitamin d", "25 - hydroxy", "25-hydroxy", "cholecalciferol"], "category": "Vitamins",
     "analogy": "Your 'sunshine' vitamin for strong bones.",
     "explanation": "Vitamin D supports bone health and immunity. Deficiency is very common."},
    {"keywords": ["vitamin b12", "b-12", "cobalamin"], "category": "Vitamins",
     "analogy": "Fuel for nerves and red-cell production.",
     "explanation": "Vitamin B12 is needed for nerves and red blood cells. Low levels can cause fatigue and anaemia."},
    {"keywords": ["calcium"], "category": "Minerals & Electrolytes",
     "analogy": "Building material for bones and a key signal for muscles.",
     "explanation": "Calcium is vital for bones, nerves, and muscle function."},
    {"keywords": ["sodium", "na+"], "category": "Minerals & Electrolytes",
     "analogy": "A salt that balances your body's water.",
     "explanation": "Sodium helps regulate fluid balance and nerve function."},
    {"keywords": ["potassium", "k+"], "category": "Minerals & Electrolytes",
     "analogy": "A mineral that keeps your heartbeat steady.",
     "explanation": "Potassium is critical for heart rhythm and muscle function; abnormal levels need attention."},

    # ---- Inflammation ----
    {"keywords": ["esr", "sedimentation"], "category": "Inflammation Markers",
     "analogy": "A general 'something's going on' smoke alarm.",
     "explanation": "ESR is a non-specific marker that rises with inflammation or infection."},
    {"keywords": ["crp", "c-reactive"], "category": "Inflammation Markers",
     "analogy": "A faster smoke alarm for inflammation.",
     "explanation": "CRP rises quickly with inflammation or infection and falls as it settles."},
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def match_test(test_name: str) -> Optional[Dict[str, Any]]:
    """Find the best knowledge-base entry for a given test name."""
    name = _norm(test_name)
    if not name:
        return None
    best = None
    best_len = 0
    for entry in KB:
        for kw in entry["keywords"]:
            # word-ish boundary match so 'hb' doesn't match inside 'thbx'
            if re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", name):
                # prefer the longest keyword hit (more specific)
                if len(kw) > best_len:
                    best = entry
                    best_len = len(kw)
    return best


# --------------------------------------------------------------------------
# Build the analysis payload
# --------------------------------------------------------------------------

def _value_with_unit(item: Dict[str, Any]) -> str:
    value = str(item.get("value", "")).strip()
    unit = str(item.get("unit", "") or "").strip()
    return f"{value} {unit}".strip()


def build_analysis(parsed_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn a parsed report into the analysis structure the frontend renders.
    Fully local — no network, no API key.
    """
    data = parsed_report.get("data", []) or []

    details: Dict[str, List[Dict[str, Any]]] = {}
    abnormal: List[Tuple[str, str]] = []  # (test_name, status)
    normal_count = 0

    for raw in data:
        test_name = str(raw.get("test_name", "")).strip()
        if not test_name:
            continue
        ref = str(raw.get("ref_interval", "") or "")
        status = compute_status(str(raw.get("value", "")), ref)

        kb = match_test(test_name)
        if kb:
            category = kb["category"]
            analogy = kb["analogy"]
            explanation = kb["explanation"]
        else:
            category = "Other Tests"
            analogy = "One of the markers measured in your report."
            explanation = (
                "This value is shown as reported by the lab. "
                "Compare it against the reference range, and ask your doctor if anything looks off."
            )

        # Add a range-aware nudge to the explanation when we judged it abnormal.
        if status in ("High", "Low"):
            explanation = f"{explanation} Your result reads {status.lower()} compared to the reference range."
            abnormal.append((test_name, status))
        elif status == "Normal":
            normal_count += 1

        details.setdefault(category, []).append({
            "test_name": test_name,
            "value": _value_with_unit(raw),
            "status": status,
            "analogy": analogy,
            "explanation": explanation,
        })

    # ---- Summary ----
    total = sum(len(v) for v in details.values())
    if total == 0:
        summary = (
            "We could not confidently read any test values from this report. "
            "Please try a clearer PDF or a text-based report."
        )
    else:
        parts = [f"We analysed {total} result{'s' if total != 1 else ''} from your report."]
        if normal_count:
            parts.append(f"{normal_count} {'are' if normal_count != 1 else 'is'} within the normal range.")
        if abnormal:
            listed = ", ".join(f"{n} ({s})" for n, s in abnormal[:6])
            more = "" if len(abnormal) <= 6 else f", and {len(abnormal) - 6} more"
            parts.append(f"A few values may need attention: {listed}{more}.")
        else:
            parts.append("Nothing stood out as clearly outside its reference range.")
        summary = " ".join(parts)

    return {
        "summary": summary,
        "details": details,
        "disclaimer": DISCLAIMER,
    }
