# parser.py
import pytesseract
import os
# Only use the Windows path if running on Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import fitz
import re
import json
from typing import List, Dict, Any, Optional
from PIL import Image, ImageOps, ImageFilter

DEBUG_DIR = "data/parser_debugs_v3"
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_debug_report(lab_name: str, filename: str, raw_text: str, parsed_results: Any, extra: Optional[dict] = None):
    base_name = os.path.basename(filename) if filename else "uploaded.pdf"
    debug_file = os.path.join(DEBUG_DIR, f"{lab_name}_{base_name}.debug.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write("==== RAW EXTRACTED TEXT ====\n\n")
        f.write(raw_text or "")
        f.write("\n\n==== PARSED RESULTS ====\n\n")
        f.write(json.dumps(parsed_results, indent=2, default=str))
        if extra:
            f.write("\n\n==== EXTRA ====\n\n")
            f.write(json.dumps(extra, indent=2, default=str))
    print(f"[DEBUG] Saved debug log for {lab_name}: {debug_file}")


def identify_lab(text: str) -> str:
    text_lower = (text or "").lower()
    if "dr" in text_lower and "lal" in text_lower and "pathlabs" in text_lower:
        return "lal_pathlabs"
    if "healthians" in text_lower and "smart report" in text_lower:
        return "healthians"
    if "apollo" in text_lower and "diagnostics" in text_lower:
        return "apollo"
    if "awadh" in text_lower and "pathology" in text_lower:
        return "awadh"
    return "unknown"

# -------------------- OCR helpers --------------------
def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = 200, lang: str = "eng") -> str:
    """
    Render each PDF page to PNG via PyMuPDF, do light preprocessing in PIL,
    then run pytesseract. Returns concatenated text for the whole PDF.
    - dpi: rendering DPI (200-300 recommended for OCR)
    - lang: tesseract language (e.g. "eng" or "eng+hin" if you have hindi traineddata)
    """
    out_text = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                # Render at requested DPI
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes(output="png")
                img = Image.open(io.BytesIO(img_bytes))

                # Light preprocessing: grayscale, autocontrast, slight sharpening, binarize
                img = img.convert("L")
                img = ImageOps.autocontrast(img)
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=2))

                # Binarize with a threshold (helps many scanned reports)
                thresh = 150
                bin_img = img.point(lambda p: 255 if p > thresh else 0)

                # run tesseract
                try:
                    page_text = pytesseract.image_to_string(bin_img, lang=lang, config="--psm 6")
                except Exception:
                    page_text = pytesseract.image_to_string(bin_img, lang="eng", config="--psm 6")
                out_text.append(page_text)
    except Exception as e:
        print(f"[WARN] ocr_pdf_bytes failed: {e}")
        return ""
    return "\n".join(out_text)


# --------------------- OCR Cleaner ---------------------
def clean_ocr_text(text: str) -> str:
    """
    Cleans common OCR mistakes and normalizes text
    before passing it to parsers.
    """
    fixes = {
        "gmvdl": "g/dL",
        "gm/dl": "g/dL",
        "icnim": "/cmm",
        "icumm": "/cumm",
        "emm": "cmm",
        "Lakhicnim": "Lakh/cmm",
        "Lakh/cnim": "Lakh/cmm",
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)

    # Replace underscores/misplaced chars with spaces
    text = text.replace("_", " ")

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # Try to merge broken lines where value + ref split into new lines
    text = re.sub(r"(\d)\s+(\d+-\d+)", r"\1 \2", text)

    return text.strip()

# --------------------- Chaos parser (OCR fallback) ---------------------
def chaos_parser(text: str) -> List[Dict[str, str]]:
    """
    Loose, forgiving parser intended for OCR'd or noisy text.
    - Finds lines that look like: TEST_NAME  VALUE  [UNIT]  [REF]
    - Returns a list of dicts: test_name, value, unit, ref_interval
    - Used when standard regex parsers fail (image-based PDFs, scanned reports).
    """
    results: List[Dict[str, str]] = []
    if not text:
        return results

    # Split into lines and attempt to match noisy patterns
    lines = text.splitlines()

    # A forgiving regex capturing a test name and a numeric-like value; units and ref ranges optional
    # Accepts µ and μ, percent, common unit characters.
    pattern = re.compile(
        r'^(?P<test>[A-Za-z][A-Za-z0-9\s\(\)\-\/\.,\%:+]{3,80}?)\s+'
        r'(?P<value>[<>]?\d{1,3}(?:[.,]\d+)?(?:\s*[/]\s*\d+)?(?:e[+-]?\d+)?)\s*'
        r'(?P<unit>[A-Za-z/%µμ\^\-0-9°·\/\[\]\(\)]+)?'
        r'(?:\s+(?P<ref>[\d\.\-<>≤≥\s%–]{2,30}))?$',
        re.UNICODE
    )

    for ln in lines:
        line = ln.strip()
        if not line or len(line) < 4:
            continue
        m = pattern.match(line)
        if m:
            test_name = (m.group("test") or "").strip()
            value = (m.group("value") or "").strip()
            unit = (m.group("unit") or "").strip()
            ref = (m.group("ref") or "").strip()

            # Basic sanitation
            test_name = re.sub(r'\s{2,}', ' ', test_name)
            test_name = re.sub(r'(^[\-\:]+)|([\-\:]+$)', '', test_name).strip()

            if test_name and value:
                results.append({
                    "test_name": test_name,
                    "value": value.replace(',', '.'),
                    "unit": unit,
                    "ref_interval": ref
                })

    # Try to salvage multi-line patterns where value is on next line
    # E.g. "Glucose\n  90 mg/dL 70-110"
    for i, ln in enumerate(lines[:-1]):
        a = ln.strip()
        b = lines[i + 1].strip()
        if not a or not b:
            continue
        # if first line looks like a test name and next line starts with a number
        if re.match(r'^[A-Za-z][A-Za-z\s\(\)\-]{3,80}$', a) and re.match(r'^[<>]?\d', b):
            combo = f"{a} {b}"
            m = pattern.match(combo)
            if m:
                test_name = (m.group("test") or "").strip()
                value = (m.group("value") or "").strip()
                unit = (m.group("unit") or "").strip()
                ref = (m.group("ref") or "").strip()
                if test_name and value:
                    results.append({
                        "test_name": test_name,
                        "value": value.replace(',', '.'),
                        "unit": unit,
                        "ref_interval": ref
                    })

    # Deduplicate preserving first seen
    unique: List[Dict[str, str]] = []
    seen = set()
    for r in results:
        key = (r.get("test_name", "").lower().strip(), r.get("value", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# ------------------ Your existing parsers (kept intact) ------------------
def parse_lal_pathlabs(text: str) -> List[Dict[str, str]]:
    """
    Enhanced parser for Dr. Lal PathLabs reports.
    Handles concatenated formats like 'g/dL12.70' and various test layouts.
    """
    results: List[Dict[str, str]] = []
    if not text:
        return results

    lines = text.split('\n')

    # Define patterns for different test result formats
    patterns = [
        # Pattern 1: Handle concatenated unit+value (e.g., "g/dL12.70")
        # This matches: test_name reference_range unit+value
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+\s*-\s*[\d\.]+)\s*([a-zA-Z/%]+)([\d\.]+)$',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(4), 'unit': m.group(3), 'ref_interval': m.group(2)}),

        # Pattern 2: Standard format with clear separation
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+)\s+([a-zA-Z/%]+)\s+([\d\.]+\s*-\s*[\d\.]+)$',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(2), 'unit': m.group(3), 'ref_interval': m.group(4)}),

        # Pattern 3: Handle percentage without explicit unit symbol
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+)\s*%\s*([\d\.]+\s*-\s*[\d\.]+)$',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(2), 'unit': '%', 'ref_interval': m.group(3)}),

        # Pattern 4: Handle complex units like thou/mm3, mill/mm3
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+)\s+(thou/mm3|mill/mm3|10\^[36]/[μu]L|fL|pg|g/dL|nmol/L|U/L|µIU/mL)\s*([\d\.]+\s*-\s*[\d\.]+)$',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(2), 'unit': m.group(3), 'ref_interval': m.group(4)}),

        # Pattern 5: Handle special cases with value first
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s*%?([\d\.]+)\s*$',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(2), 'unit': '', 'ref_interval': ''}),
    ]

    # Try to find the test results section
    in_results_section = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for section markers
        if 'Test Name' in line and 'Results' in line and 'Bio. Ref. Interval' in line:
            in_results_section = True
            i += 1
            continue

        # Stop at comment sections
        if any(keyword in line.lower() for keyword in ['comment', 'interpretation', 'note', '------']):
            in_results_section = False

        if in_results_section and line:
            # Try each pattern
            matched = False
            for pattern, extractor in patterns:
                match = re.match(pattern, line)
                if match:
                    result = extractor(match)
                    # Clean up test name
                    result['test_name'] = re.sub(r'^\s*-?\s*', '', result['test_name'])
                    result['test_name'] = re.sub(r'\s+', ' ', result['test_name']).strip()

                    if result['test_name'] and len(result['test_name']) > 2:
                        results.append(result)
                        matched = True
                        break

            # Special handling for multi-line entries
            if not matched and i + 1 < len(lines):
                # Check if next line contains unit and value
                next_line = lines[i + 1].strip()
                combined = f"{line} {next_line}"

                for pattern, extractor in patterns:
                    match = re.match(pattern, combined)
                    if match:
                        result = extractor(match)
                        result['test_name'] = re.sub(r'^\s*-?\s*', '', result['test_name'])
                        result['test_name'] = re.sub(r'\s+', ' ', result['test_name']).strip()

                        if result['test_name'] and len(result['test_name']) > 2:
                            results.append(result)
                            i += 1  # Skip next line since we used it
                            break

        i += 1

    # Additional targeted extraction for specific tests that might be missed
    specific_tests = {
        r'Hemoglobin[^\n]*?([\d\.]+)\s*g/dL\s*([\d\.]+\s*-\s*[\d\.]+)':
            lambda m: {'test_name': 'Hemoglobin', 'value': m.group(1), 'unit': 'g/dL', 'ref_interval': m.group(2)},
        r'Packed Cell Volume[^\n]*?([\d\.]+)\s*%\s*([\d\.]+\s*-\s*[\d\.]+)':
            lambda m: {'test_name': 'Packed Cell Volume (PCV)', 'value': m.group(1), 'unit': '%', 'ref_interval': m.group(2)},
        r'RBC Count[^\n]*?([\d\.]+)\s*mill/mm3\s*([\d\.]+\s*-\s*[\d\.]+)':
            lambda m: {'test_name': 'RBC Count', 'value': m.group(1), 'unit': 'mill/mm3', 'ref_interval': m.group(2)},
        r'Total Leukocyte Count[^\n]*?([\d\.]+)\s*thou/mm3\s*([\d\.]+\s*-\s*[\d\.]+)':
            lambda m: {'test_name': 'Total Leukocyte Count (TLC)', 'value': m.group(1), 'unit': 'thou/mm3', 'ref_interval': m.group(2)},
        r'Platelet Count[^\n]*?([\d\.]+)\s*thou/mm3\s*([\d\.]+\s*-\s*[\d\.]+)':
            lambda m: {'test_name': 'Platelet Count', 'value': m.group(1), 'unit': 'thou/mm3', 'ref_interval': m.group(2)},
        r'VITAMIN D[^\n]*?([\d\.]+|<[\d\.]+)\s*nmol/L\s*([\d\.]+\s*-\s*[\d\.]+)':
            lambda m: {'test_name': 'VITAMIN D, 25 - HYDROXY', 'value': m.group(1), 'unit': 'nmol/L', 'ref_interval': m.group(2)},
        r'ALT \(SGPT\)[^\n]*?([\d\.]+)\s*U/L\s*(<?\s*[\d\.]+)':
            lambda m: {'test_name': 'ALT (SGPT)', 'value': m.group(1), 'unit': 'U/L', 'ref_interval': m.group(2)},
    }

    # Try specific test patterns on the entire text
    for pattern, extractor in specific_tests.items():
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            result = extractor(match)
            # Check if we already have this test
            exists = any(r['test_name'].lower() == result['test_name'].lower()
                        and r['value'] == result['value'] for r in results)
            if not exists:
                results.append(result)

    # Handle special concatenated format more aggressively
    concat_pattern = r'([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+\s*-\s*[\d\.]+)\s*\n\s*([a-zA-Z/%]+)([\d\.]+)'
    concat_matches = re.finditer(concat_pattern, text, re.MULTILINE)
    for match in concat_matches:
        result = {
            'test_name': match.group(1).strip(),
            'value': match.group(4),
            'unit': match.group(3),
            'ref_interval': match.group(2)
        }
        # Check if we already have this test
        exists = any(r['test_name'].lower() == result['test_name'].lower()
                    and r['value'] == result['value'] for r in results)
        if not exists and result['test_name'] and len(result['test_name']) > 2:
            results.append(result)

    # Remove duplicates
    unique_results = []
    seen = set()
    for r in results:
        key = (r.get('test_name', '').lower().strip(), r.get('value', ''))
        if key not in seen and r.get('test_name') and r.get('value'):
            seen.add(key)
            unique_results.append(r)

    return unique_results


def parse_apollo(text: str) -> List[Dict[str, str]]:
    """
    Parser for Apollo Diagnostics reports.
    """
    results: List[Dict[str, str]] = []
    if not text:
        return results

    # Pattern for Apollo format
    patterns = [
        # Standard Apollo format
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+)\s+([a-zA-Z\dµ/%]+)\s+([\d\.]+\s*-\s*[\d\.]+)',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(2), 'unit': m.group(3), 'ref_interval': m.group(4)}),

        # Format with Bio. Ref. Interval
        (r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\s+([\d\.]+)\s+([a-zA-Z\dµ/%]+)\s+([\d\.]+\s*-\s*[\d\.]+)\s+\w+$',
         lambda m: {'test_name': m.group(1).strip(), 'value': m.group(2), 'unit': m.group(3), 'ref_interval': m.group(4)}),
    ]

    lines = text.split('\n')
    in_results_section = False

    for line in lines:
        line = line.strip()

        # Look for test result markers
        if 'Test Name' in line and 'Result' in line:
            in_results_section = True
            continue

        if 'Comment' in line or '***' in line:
            in_results_section = False

        if in_results_section and line:
            for pattern, extractor in patterns:
                match = re.match(pattern, line)
                if match:
                    result = extractor(match)
                    if result['test_name'] and len(result['test_name']) > 2:
                        # Filter out headers and invalid entries
                        if not any(k in result['test_name'].lower() for k in ["order id", "age", "reported", "sample", "patient"]):
                            results.append(result)
                    break

    return results


def parse_healthians(text: str) -> List[Dict[str, str]]:
    """
    Parser for Healthians Smart Report format.
    """
    results: List[Dict[str, str]] = []
    if not text:
        return results

    # Healthians has a specific format with Method: lines
    pattern = re.compile(
        r'^([A-Za-z][A-Za-z\s\(\),\-]+?)\n'    # Test Name
        r'Method:\s*.*?\n'                      # Method line
        r'(?:Machine:\s*.*?\n)?'                # Optional Machine
        r'([\d\.]+)\s+'                         # Value
        r'([a-zA-Z\d\^/%µ]+)\s+'               # Unit
        r'([\d\.\s\-<>]+)',                     # Reference range
        re.MULTILINE
    )

    matches = pattern.findall(text)
    for match in matches:
        test_name = match[0].replace('\n', ' ').strip()
        results.append({
            "test_name": test_name,
            "value": match[1].strip(),
            "unit": match[2].strip(),
            "ref_interval": match[3].strip()
        })

    # Also try line-by-line parsing for Healthians
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this looks like a test name
        if line and not line.startswith('Method:') and not line.startswith('Machine:'):
            # Check if next lines contain Method: and values
            if i + 2 < len(lines) and 'Method:' in lines[i + 1]:
                test_name = line
                # Skip Method and Machine lines
                j = i + 1
                while j < len(lines) and ('Method:' in lines[j] or 'Machine:' in lines[j]):
                    j += 1

                # Now try to extract value, unit, ref_interval
                if j < len(lines):
                    value_line = lines[j].strip()
                    # Pattern: value unit ref_interval
                    value_match = re.match(r'([\d\.]+)\s+([a-zA-Z\d\^/%µ]+)\s+([\d\.\s\-<>]+)', value_line)
                    if value_match:
                        results.append({
                            "test_name": test_name,
                            "value": value_match.group(1),
                            "unit": value_match.group(2),
                            "ref_interval": value_match.group(3)
                        })
                        i = j
        i += 1

    # Remove duplicates
    unique_results = []
    seen = set()
    for r in results:
        key = (r.get('test_name', '').lower().strip(), r.get('value', ''))
        if key not in seen and r.get('test_name') and r.get('value'):
            seen.add(key)
            unique_results.append(r)

    return unique_results


# ------------------- helpers for OCR cleanup -------------------
def _ocr_text_cleanup(text: str) -> str:
    """
    Fix common OCR misreads that often appear in scanned reports.
    Very light touch: numbers and symbols only.
    """
    if not text:
        return ""

    replacements = {
        "O": "0",    # O -> zero
        "o": "0",
        "I": "1",    # capital I -> 1
        "l": "1",    # lowercase L -> 1
        "—": "-",    # em dash -> hyphen
        "–": "-",    # en dash -> hyphen
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


# ------------------- Awadh parser (uses chaos + cleanup) -------------------
def parse_awadh(text: str) -> List[Dict[str, str]]:
    """
    Parser for Awadh Pathology reports.
    - Cleans up OCR'd text
    - Runs chaos_parser (for noisy OCR)
    - If chaos_parser yields too few results, log a note
    """
    cleaned = _ocr_text_cleanup(text)

    results = chaos_parser(cleaned)

    if len(results) < 3:
        print("[INFO] Awadh parser: very few results from chaos_parser")
    return results

# -------------------- CLI / quick test runner --------------------
def main():
    data_folder = 'data'
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    all_reports_data = {}

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"--- Processing Report: {pdf_path} ---")
        full_text = ""

        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    full_text += page.get_text()

            lab_name = identify_lab(full_text)
            print(f"Identified Lab: {lab_name}")

            parsed_data = []
            if lab_name == "lal_pathlabs":
                parsed_data = parse_lal_pathlabs(full_text)
            elif lab_name == "apollo":
                parsed_data = parse_apollo(full_text)
            elif lab_name == "healthians":
                parsed_data = parse_healthians(full_text)
            elif lab_name == "awadh":
                parsed_data = parse_awadh(full_text)

            all_reports_data[pdf_path] = parsed_data

            # Save debug report
            save_debug_report(lab_name, pdf_file, full_text, parsed_data)

        except Exception as e:
            all_reports_data[pdf_path] = {"error": str(e)}
            print(f"Error processing {pdf_file}: {str(e)}")

    print(json.dumps(all_reports_data, indent=2))

# Export the enhanced parser for use in main.py
enhanced_parse_lal_pathlabs = parse_lal_pathlabs
if __name__ == "__main__":
    main()

