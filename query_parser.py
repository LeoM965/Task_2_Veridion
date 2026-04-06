import re

# Region code sets

EUROPEAN_COUNTRY_CODES = {
    "al", "ad", "at", "by", "be", "ba", "bg", "hr", "cy", "cz", "dk",
    "ee", "fi", "fr", "de", "gr", "hu", "is", "ie", "it", "lv", "li",
    "lt", "lu", "mt", "md", "me", "nl", "mk", "no", "pl", "pt", "ro",
    "rs", "sk", "si", "es", "se", "ch", "ua", "gb",
}

SCANDINAVIAN_COUNTRY_CODES = {"se", "no", "dk", "fi", "is"}

# Maps location keywords found in the query → allowed country codes
LOCATION_KEYWORDS = {
    "romania":       {"ro"},
    "france":        {"fr"},
    "germany":       {"de"},
    "united states": {"us"},
    "switzerland":   {"ch"},
    "scandinavia":   SCANDINAVIAN_COUNTRY_CODES,
    "europe":        EUROPEAN_COUNTRY_CODES,
}

# Synonym expansion
# When a keyword appears in the query, these related terms are appended
# to improve recall (e.g. "logistics" also searches for "freight", "shipping", etc.)

SYNONYM_MAP = {
    "logistics":        ["freight", "shipping", "transportation", "warehousing",
                         "supply chain", "forwarding", "distribution", "3pl"],
    "software":         ["saas", "platform", "cloud", "application", "digital"],
    "construction":     ["building", "civil engineering", "contractor",
                         "infrastructure", "general contractor"],
    "pharmaceutical":   ["pharma", "drug", "biotech", "medicine", "clinical",
                         "therapeutics"],
    "fintech":          ["financial technology", "payments", "neobank",
                         "digital banking", "lending platform"],
    "e-commerce":       ["ecommerce", "online retail", "online store",
                         "marketplace", "shopify", "direct-to-consumer"],
    "renewable energy": ["solar", "wind", "clean energy", "green energy",
                         "photovoltaic", "wind turbine"],
    "clean energy":     ["solar", "wind", "renewable", "green energy",
                         "photovoltaic", "wind turbine"],
    "packaging":        ["package", "container", "wrapping", "carton",
                         "label", "bottle", "box"],
    "food":             ["beverage", "dairy", "meat", "bakery", "snack",
                         "grocery", "confectionery"],
    "hr":               ["human resources", "payroll", "recruitment",
                         "talent", "workforce", "staffing"],
    "electric vehicle": ["ev", "battery", "lithium", "cathode", "anode",
                         "cell", "electrolyte", "bms"],
    "battery":          ["lithium", "cell", "cathode", "anode",
                         "electrolyte", "energy storage"],
}

# NAICS industry code prefixes
# Maps query keywords → NAICS code prefixes that indicate a direct industry match

NAICS_PREFIX_MAP = {
    "logistics":        ["488", "493", "484"],
    "software":         ["511", "518", "541"],
    "construction":     ["236", "237", "238"],
    "pharmaceutical":   ["325", "424"],
    "food":             ["311", "312"],
    "packaging":        ["322", "326"],
    "renewable energy": ["221", "333", "335"],
    "clean energy":     ["221", "333", "335"],
    "fintech":          ["522", "523", "524"],
    "e-commerce":       ["454"],
    "hr":               ["541", "561"],
    "electric vehicle": ["335", "336", "325"],
    "battery":          ["335", "325"],
}

# Words that indicate the user is looking for a supplier/manufacturer
SUPPLIER_ROLE_KEYWORDS = [
    "supply", "supplier", "manufacture", "manufacturer",
    "produce", "producer", "component", "provide",
]


def expand_query_with_synonyms(query):
    """Return the query enriched with synonym terms for better text matching."""
    query_lower = query.lower()
    extra_terms = []

    for keyword, related_terms in SYNONYM_MAP.items():
        if keyword in query_lower:
            extra_terms.extend(related_terms)

    if extra_terms:
        return query + " " + " ".join(extra_terms)
    return query


def find_naics_prefixes_for_query(query):
    """Return NAICS code prefixes relevant to keywords found in the query."""
    query_lower = query.lower()
    matched_prefixes = []

    for keyword, prefixes in NAICS_PREFIX_MAP.items():
        if keyword in query_lower:
            matched_prefixes.extend(prefixes)

    return list(set(matched_prefixes))


def parse_query(query):
    """
    Extract structured filters from a natural language query.

    Returns a dict with possible keys:
        geo           → set of allowed country codes
        is_public     → True if user wants public companies
        min_emp       → minimum employee count
        max_emp       → maximum employee count
        min_rev       → minimum revenue in USD
        min_year      → company must be founded after this year
        b2b           → True if B2B business model required
        saas          → True if SaaS business model required
        naics_prefixes → list of NAICS code prefixes for industry matching
        role          → "supplier" if the query looks for suppliers/manufacturers
    """
    query_lower = query.lower()
    filters = {}

    # location filter
    for location_name, country_codes in LOCATION_KEYWORDS.items():
        if location_name in query_lower:
            filters["geo"] = country_codes

    # public companies only
    if "public " in query_lower:
        filters["is_public"] = True

    # employee count bounds
    match = re.search(r"more than ([\d,]+) employees", query_lower)
    if match:
        filters["min_emp"] = int(match.group(1).replace(",", ""))

    match = re.search(r"fewer than ([\d,]+) employees", query_lower)
    if match:
        filters["max_emp"] = int(match.group(1).replace(",", ""))

    # revenue floor
    match = re.search(r"revenue over \$?([\d,]+)\s*million", query_lower)
    if match:
        filters["min_rev"] = float(match.group(1).replace(",", "")) * 1_000_000

    # founding year
    match = re.search(r"after (\d{4})", query_lower)
    if match:
        filters["min_year"] = int(match.group(1))

    # business model flags
    if "b2b" in query_lower:
        filters["b2b"] = True
    if "saas" in query_lower:
        filters["saas"] = True

    # NAICS industry match
    naics_prefixes = find_naics_prefixes_for_query(query)
    if naics_prefixes:
        filters["naics_prefixes"] = naics_prefixes

    # supply chain role detection
    for word in SUPPLIER_ROLE_KEYWORDS:
        if word in query_lower:
            filters["role"] = "supplier"
            break

    return filters
