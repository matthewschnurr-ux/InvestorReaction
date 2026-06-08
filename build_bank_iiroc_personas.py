"""Build per-firm advisor persona JSON files from the enriched spreadsheet.

Reads the Full_Service_Broker_Advisor_Personas_enriched.xlsx workbook and
produces one JSON file per firm in the schema used by rbc_ds_personas.json,
so the app's existing advisor code paths (attach_advisor_metadata,
stratified_sample_rbc, demographics expander) work unchanged.
"""
import json
import os
import re
import pandas as pd

SOURCE = r"C:\Users\mschn\OneDrive\Documents\New project\outputs\advisor-research\Full_Service_Broker_Advisor_Personas_enriched.xlsx"
OUT_DIR = os.path.dirname(__file__)

FIRMS = {
    "BMO Nesbitt Burns":          ("BMO Nesbitt Burns",          "bmo_nesbitt_burns_personas.json"),
    "CIBC Wood Gundy":            ("CIBC Wood Gundy",            "cibc_wood_gundy_personas.json"),
    "TD Wealth":                  ("TD Wealth",                  "td_wealth_personas.json"),
    "National Bank Financial WM": ("National Bank Financial WM", "nbf_wm_personas.json"),
    "Scotia Wealth":              ("Scotia Wealth",              "scotia_wealth_personas.json"),
}

PROV_CODE_TO_NAME = {
    "ON": "Ontario", "QC": "Quebec", "BC": "British Columbia", "AB": "Alberta",
    "MB": "Manitoba", "SK": "Saskatchewan", "NS": "Nova Scotia",
    "NB": "New Brunswick", "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island", "YT": "Yukon", "NT": "Northwest Territories",
    "NU": "Nunavut",
}
# Match either a 2-letter code OR a full province name.
_FULL_PROV_NAMES = [
    "British Columbia", "Newfoundland and Labrador", "Northwest Territories",
    "Nova Scotia", "New Brunswick", "Prince Edward Island",
    "Ontario", "Quebec", "Alberta", "Manitoba", "Saskatchewan", "Yukon", "Nunavut",
]
_NAME_TO_CANON = {n.lower(): n for n in _FULL_PROV_NAMES}
PROV_RE_CODE = re.compile(r"\b(" + "|".join(PROV_CODE_TO_NAME.keys()) + r")\b")
PROV_RE_NAME = re.compile(r"\b(" + "|".join(re.escape(n) for n in _FULL_PROV_NAMES) + r")\b", re.IGNORECASE)


# City -> province fallback for rows where no province text appears anywhere.
# Covers the major Canadian financial-services markets.
CITY_TO_PROVINCE = {
    # Ontario
    "toronto": "Ontario", "ottawa": "Ontario", "mississauga": "Ontario",
    "brampton": "Ontario", "hamilton": "Ontario", "london": "Ontario",
    "markham": "Ontario", "vaughan": "Ontario", "kitchener": "Ontario",
    "windsor": "Ontario", "richmond hill": "Ontario", "oakville": "Ontario",
    "burlington": "Ontario", "barrie": "Ontario", "sudbury": "Ontario",
    "kingston": "Ontario", "guelph": "Ontario", "thornhill": "Ontario",
    "north york": "Ontario", "scarborough": "Ontario", "etobicoke": "Ontario",
    "waterloo": "Ontario", "cambridge": "Ontario", "st. catharines": "Ontario",
    "niagara falls": "Ontario", "pickering": "Ontario", "ajax": "Ontario",
    "whitby": "Ontario", "oshawa": "Ontario", "milton": "Ontario",
    "newmarket": "Ontario", "woodstock": "Ontario", "sarnia": "Ontario",
    "peterborough": "Ontario", "belleville": "Ontario", "thunder bay": "Ontario",
    "chatham": "Ontario",
    # Quebec
    "montreal": "Quebec", "montréal": "Quebec", "quebec city": "Quebec",
    "laval": "Quebec", "gatineau": "Quebec", "longueuil": "Quebec",
    "sherbrooke": "Quebec", "trois-rivières": "Quebec", "kirkland": "Quebec",
    "saint-georges": "Quebec",
    # British Columbia
    "vancouver": "British Columbia", "surrey": "British Columbia",
    "burnaby": "British Columbia", "richmond": "British Columbia",
    "victoria": "British Columbia", "kelowna": "British Columbia",
    "abbotsford": "British Columbia", "kamloops": "British Columbia",
    "nanaimo": "British Columbia", "west vancouver": "British Columbia",
    "north vancouver": "British Columbia", "coquitlam": "British Columbia",
    "langley": "British Columbia", "white rock": "British Columbia",
    # Alberta
    "calgary": "Alberta", "edmonton": "Alberta", "red deer": "Alberta",
    "lethbridge": "Alberta", "medicine hat": "Alberta", "grande prairie": "Alberta",
    "st. albert": "Alberta", "okotoks": "Alberta",
    # Manitoba
    "winnipeg": "Manitoba", "brandon": "Manitoba",
    # Saskatchewan
    "saskatoon": "Saskatchewan", "regina": "Saskatchewan",
    # Nova Scotia
    "halifax": "Nova Scotia", "dartmouth": "Nova Scotia", "wolfville": "Nova Scotia",
    # New Brunswick
    "saint john": "New Brunswick", "moncton": "New Brunswick",
    "fredericton": "New Brunswick",
    # Newfoundland
    "st. john's": "Newfoundland and Labrador",
    # PEI
    "charlottetown": "Prince Edward Island",
}


def _province_from_city(city):
    return CITY_TO_PROVINCE.get(_s(city).lower())


def _province_from_text(s):
    """Return canonical province name found in text, or None."""
    if not s:
        return None
    m = PROV_RE_NAME.search(s)
    if m:
        return _NAME_TO_CANON[m.group(1).lower()]
    m = PROV_RE_CODE.search(s)
    if m:
        return PROV_CODE_TO_NAME[m.group(1)]
    return None


def _s(v):
    """String-or-empty helper that treats NaN as empty."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _int(v, default=0):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _split_name(full):
    full = _s(full)
    if not full:
        return "", ""
    parts = full.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _parse_city_province(loc, summary_fallback="", branch_fallback=""):
    """'Thornhill, ON' or 'Barrie, Ontario' -> ('Thornhill', 'Ontario').
    Falls back to scanning the branch address and persona-summary text when
    the Location column doesn't include a province."""
    loc = _s(loc)
    summary_fallback = _s(summary_fallback)
    branch_fallback = _s(branch_fallback)

    # Province: Location → Branch address → Summary text → city lookup.
    province = (_province_from_text(loc)
                or _province_from_text(branch_fallback)
                or _province_from_text(summary_fallback)
                or "Unknown")

    # City for the lookup fallback: prefer the comma-prefix of Location.
    if province == "Unknown":
        candidate_city = loc.split(",", 1)[0].strip() if "," in loc else loc.strip()
        guess = _province_from_city(candidate_city)
        if guess:
            province = guess

    # City: everything before the first comma in Location; else the field itself.
    if "," in loc:
        city = loc.split(",", 1)[0].strip()
    elif loc:
        city = loc.strip()
    else:
        city = "Unknown"

    # If the "city" is actually a known province name or code, blank it out.
    if city.lower() in _NAME_TO_CANON or city.upper() in PROV_CODE_TO_NAME:
        city = "Unknown"
    # Filter out obviously non-city junk (e.g. raw area codes like '403').
    if city.isdigit():
        city = "Unknown"
    return city or "Unknown", province


def _parse_designations(s):
    s = _s(s)
    if not s:
        return []
    # Split on common separators
    parts = re.split(r"[,;/|]| and ", s)
    return [p.strip() for p in parts if p.strip()]


def _parse_pim(v):
    s = _s(v).upper()
    if s.startswith("Y"):
        return "Y"
    if s.startswith("N"):
        return "N"
    return "Unknown"


def _build_persona(row, firm_label, persona_id):
    first, last = _split_name(row.get("Name"))
    city, province = _parse_city_province(
        row.get("Location (City)"),
        summary_fallback=row.get("Persona Summary"),
        branch_fallback=row.get("Branch / Office"),
    )
    years = _int(row.get("Years in Industry"), default=15)
    designations = _parse_designations(row.get("Designations / Licensing"))
    pim = _parse_pim(row.get("PIM / Discretionary (Y/N)"))
    practice_focus = _s(row.get("Products / Services Emphasized")) or "Wealth Management"
    title = _s(row.get("Title")) or "Investment Advisor"
    team_size = max(1, _int(row.get("Team Size (people)"), default=1))
    summary = _s(row.get("Persona Summary"))
    if not summary:
        summary = f"{first} {last} is a financial advisor at {firm_label}. Title: {title}. " \
                  f"Approximately {years} years in the industry (experience estimate)."
    education = _s(row.get("Education")) or "Unknown"
    target_clientele = _s(row.get("Target Clientele"))
    client_demographics = target_clientele or "High net worth clients"

    return {
        "id": persona_id,
        "first_name": first,
        "last_name": last,
        "age": 40,                       # Unknown — mirror RBC default
        "gender": "Unknown",
        "ethnicity": "Unknown",
        "province": province,
        "city": city,
        "firm_type": firm_label,
        "years_in_business": years,
        "designations": designations,
        "book_size_aum": 0,
        "num_clients": 0,
        "practice_focus": practice_focus,
        "compensation_model": "Fee-based",
        "personal_income": 0,
        "business_maturity": "Established" if years >= 16 else ("Growing" if years >= 6 else "Early"),
        "client_demographics": client_demographics,
        "team_size": team_size,
        "tech_adoption": "Medium",
        "professional_challenges": [],
        "professional_values": [],
        "title": title,
        "pim_licensed": pim,
        "education": education,
        "family_status": "Unknown",
        "persona_summary": summary,
    }


def main():
    print(f"Reading {SOURCE}")
    for sheet, (firm_label, out_name) in FIRMS.items():
        df = pd.read_excel(SOURCE, sheet_name=sheet)
        df = df[df["Name"].apply(_s) != ""]   # drop rows with no name
        personas = [_build_persona(r, firm_label, i + 1) for i, r in enumerate(df.to_dict("records"))]
        # Drop duplicates by (first, last, city) — some sheets had near-dups
        seen = set()
        deduped = []
        for p in personas:
            key = (p["first_name"].lower(), p["last_name"].lower(), p["city"].lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        # Re-id sequentially after dedup
        for i, p in enumerate(deduped):
            p["id"] = i + 1
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)
        print(f"  {firm_label:30s} -> {out_name}  ({len(deduped)} personas, {len(personas) - len(deduped)} dups dropped)")


if __name__ == "__main__":
    main()
