import fitz
import re
import json
import os

def identify_lab(text):
    text_lower = text.lower()
    if "dr" in text_lower and "lal" in text_lower and "pathlabs" in text_lower:
        return "lal_pathlabs"
    if "healthians" in text_lower and "smart report" in text_lower:
        return "healthians"
    if "apollo" in text_lower and "diagnostics" in text_lower:
        return "apollo"
    if "awadh" in text_lower and "pathology" in text_lower:
        return "awadh"
    return "unknown"

def parse_lal_pathlabs(text):
    results = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    KNOWN_UNITS = {"g/dL", "%", "mill/mm3", "fL", "pg", "thou/mm3", "nmol/L", "U/L"}
    def is_value(s): return re.fullmatch(r"[\d\.<>]+", s) is not None
    def is_range(s): return '-' in s and re.search(r"\d", s)
    def is_unit(s): return s in KNOWN_UNITS

    try:
        # Flexible header detection
        start_index = next(
            (i for i, line in enumerate(lines) if re.search(r"Bio\.?\s*Ref\.?\s*Interval", line, re.I)),
            None
        )
        if start_index is None:
            return results

        # Find stopping point (first heading-like line after start)
        stop_keywords = {"comment", "clinical", "interpretation"}
        end_index = None
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip().lower() in stop_keywords:
                end_index = i
                break
        if end_index is None:
            end_index = len(lines)

        table_tokens = lines[start_index + 1:end_index]
        for i in range(0, len(table_tokens), 4):
            chunk = table_tokens[i:i+4]
            if len(chunk) == 4:
                identified_parts, name_parts = {}, []
                for part in chunk:
                    if is_range(part):
                        identified_parts['ref_interval'] = part
                    elif is_unit(part):
                        identified_parts['unit'] = part
                    elif is_value(part):
                        identified_parts['value'] = part
                    else:
                        name_parts.append(part)
                if {'value', 'unit', 'ref_interval'} <= identified_parts.keys():
                    test_name = " ".join(name_parts).strip(" -:;")
                    if test_name:
                        identified_parts['test_name'] = test_name
                        results.append(identified_parts)
    except Exception:
        pass

    # Ensure unique
    unique_results = []
    seen = set()
    for r in results:
        sig = (r['test_name'], r['value'])
        if sig not in seen:
            seen.add(sig)
            unique_results.append(r)
    return unique_results

def parse_apollo(text):
    results = []
    pattern = re.compile(
        r"^(.*?)\n"      # Test name
        r"([\d\.]+)\n"   # Value
        r"([a-zA-Z\dµ/]+)\n"  # Unit
        r"(.*?)\n"       # Ref interval
        r"(?:[a-zA-Z\d-]+)?$",  # Optional trailing token
        re.MULTILINE
    )
    matches = pattern.findall(text)
    for match in matches:
        test_name = ' '.join(match[0].strip().split('\n')).strip(" -:;")
        # Filter junk rows
        if len(test_name) < 3 or test_name.isupper() or re.match(r"^\d", test_name):
            continue
        if any(k in test_name.lower() for k in ["order id", "age", "reported", "sample", "patient"]):
            continue
        results.append({
            "test_name": test_name,
            "value": match[1].strip(),
            "unit": match[2].strip(),
            "ref_interval": match[3].strip() if match[3].strip() else "N/A"
        })
    return results

def parse_healthians(text):
    results = []
    pattern = re.compile(
        r"^(.*?)\n"                             # Test Name
        r"Method:.*?\n"                         # Method line
        r"(?:Machine:.*?\n)?"                   # Optional Machine
        r"([\d\.]+)\n"                         # Value
        r"([a-zA-Z\d\^/µl]+)\n"                 # Unit
        r"([\d\.\s-]+)$",                       # Ref interval
        re.MULTILINE
    )
    matches = pattern.findall(text)
    for match in matches:
        test_name = match[0].replace('\n', ' ').strip(" -:;")
        results.append({
            "test_name": test_name,
            "value": match[1].strip(),
            "unit": match[2].strip(),
            "ref_interval": match[3].strip()
        })
    return results

def parse_awadh(text):
    return []

def main():
    data_folder = 'data'
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    all_reports_data = {}

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"--- Processing Report: {pdf_path} ---")
        try:
            full_text = ""
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

            all_reports_data[pdf_path] = parsed_data
        except Exception as e:
            all_reports_data[pdf_path] = {"error": str(e)}
        print("\n")

    print("--- All Parsed Data (JSON Output) ---")
    print(json.dumps(all_reports_data, indent=2))

if __name__ == "__main__":
    main()
