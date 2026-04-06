import json
import ast


def load_companies(file_path):
    """Load company records from a JSONL file and normalize messy fields."""
    companies = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                company = json.loads(line)
            except json.JSONDecodeError:
                continue

            # some fields arrive as strings instead of dicts/lists - fix them
            for field_name in ("address", "primary_naics", "secondary_naics"):
                field_value = company.get(field_name)
                if isinstance(field_value, str):
                    try:
                        company[field_name] = ast.literal_eval(field_value)
                    except Exception:
                        company[field_name] = None

            companies.append(company)

    return companies


def get_country_code(company):
    """Extract the 2-letter country code from a company's address. Returns '' if missing."""
    address = company.get("address")
    if isinstance(address, dict):
        return (address.get("country_code") or "").lower()
    return ""


def _extract_naics_entries(company):
    """Yield (code, label) pairs from both primary and secondary NAICS fields."""
    primary = company.get("primary_naics")
    if isinstance(primary, dict):
        yield primary.get("code", ""), primary.get("label", "")

    secondary = company.get("secondary_naics")
    if isinstance(secondary, dict):
        yield secondary.get("code", ""), secondary.get("label", "")
    elif isinstance(secondary, list):
        for entry in secondary:
            if isinstance(entry, dict):
                yield entry.get("code", ""), entry.get("label", "")


def get_naics_codes(company):
    """Return a list of NAICS code strings for the company."""
    return [str(code) for code, label in _extract_naics_entries(company) if code]


def get_naics_labels(company):
    """Return a list of NAICS label strings for the company."""
    return [label for code, label in _extract_naics_entries(company) if label]


def build_company_text(company):
    """Merge all relevant text fields into one searchable string (lowercased)."""
    parts = [
        company.get("operational_name", ""),
        company.get("description", ""),
    ]
    parts += company.get("core_offerings", []) or []
    parts += company.get("target_markets", []) or []
    parts += company.get("business_model", []) or []
    parts += get_naics_labels(company)

    return " ".join(str(part) for part in parts).lower()
