import re
import math
from collections import Counter
from data_loader import get_country_code, get_naics_codes

# --- Scoring weights ---
# These control how much each signal contributes to the final score.

TEXT_SIMILARITY_WEIGHT  = 0.50   # TF-IDF cosine similarity
NAICS_MATCH_WEIGHT     = 0.20   # NAICS industry code match
OFFERINGS_MATCH_WEIGHT = 0.15   # core_offerings keyword overlap
SUPPLY_ROLE_WEIGHT     = 0.15   # supply chain role relevance


def tokenize(text):
    """Split text into lowercase tokens of 3+ characters."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2]


# --- TF-IDF engine ---

def build_tfidf_index(document_texts):
    """
    Build a TF-IDF index from a list of document text strings.

    Returns:
        document_tokens: list of token lists (one per document)
        inverse_doc_freq: dict mapping each word to its IDF value
    """
    total_documents = len(document_texts)
    word_document_count = Counter()
    document_tokens = []

    for text in document_texts:
        tokens = tokenize(text)
        document_tokens.append(tokens)
        unique_words = set(tokens)
        for word in unique_words:
            word_document_count[word] += 1

    inverse_doc_freq = {
        word: math.log(total_documents / count)
        for word, count in word_document_count.items()
    }

    return document_tokens, inverse_doc_freq


def compute_text_similarity(query_text, document_tokens, inverse_doc_freq):
    """
    Compute TF-IDF cosine similarity between the query and every document.

    Returns a list of similarity scores (one per document).
    """
    query_tokens = tokenize(query_text)
    query_word_counts = Counter(query_tokens)

    # build query TF-IDF vector
    query_vector = {}
    for word, count in query_word_counts.items():
        if word in inverse_doc_freq:
            query_vector[word] = count * inverse_doc_freq[word]

    if not query_vector:
        return [0.0] * len(document_tokens)

    query_norm = math.sqrt(sum(value ** 2 for value in query_vector.values()))

    similarity_scores = []
    for tokens in document_tokens:
        doc_word_counts = Counter(tokens)

        # dot product between query vector and document vector
        dot_product = 0.0
        for word in query_vector:
            if word in doc_word_counts:
                doc_weight = doc_word_counts[word] * inverse_doc_freq.get(word, 0)
                dot_product += query_vector[word] * doc_weight

        # document vector norm
        doc_norm_squared = sum(
            (count * inverse_doc_freq.get(word, 0)) ** 2
            for word, count in doc_word_counts.items()
        )
        doc_norm = math.sqrt(doc_norm_squared) if doc_norm_squared > 0 else 1.0

        # cosine similarity
        denominator = query_norm * doc_norm
        similarity = dot_product / denominator if denominator > 0 else 0.0
        similarity_scores.append(similarity)

    return similarity_scores


# --- Signal scorers ---

def compute_naics_match(company, relevant_prefixes):
    """Return 1.0 if any company NAICS code starts with a relevant prefix, else 0.0."""
    if not relevant_prefixes:
        return 0.0

    company_naics_codes = get_naics_codes(company)
    for code in company_naics_codes:
        for prefix in relevant_prefixes:
            if code.startswith(prefix):
                return 1.0
    return 0.0


def compute_offerings_overlap(company, query_text):
    """Measure how much the company's core_offerings overlap with query keywords."""
    offerings_list = company.get("core_offerings") or []
    if not offerings_list:
        return 0.0

    query_keywords = set(tokenize(query_text))
    if not query_keywords:
        return 0.0

    offerings_text = " ".join(offering.lower() for offering in offerings_list)
    offerings_keywords = set(tokenize(offerings_text))
    if not offerings_keywords:
        return 0.0

    matching_keywords = query_keywords & offerings_keywords
    return len(matching_keywords) / len(query_keywords)


def compute_supplier_role_score(company, role):
    """Score how strongly a company signals being a supplier/manufacturer."""
    if not role:
        return 0.0

    description = (company.get("description") or "").lower()
    offerings = " ".join(company.get("core_offerings") or []).lower()
    combined_text = description + " " + offerings

    supplier_signals = [
        "manufacturer", "supplier", "producer", "fabricat",
        "supply", "component", "raw material", "wholesale",
    ]
    signal_hits = sum(1 for signal in supplier_signals if signal in combined_text)

    # cap at 1.0 — a company with 3+ supplier signals gets maximum score
    return min(signal_hits / 3.0, 1.0)


# --- Constraint penalties ---

def compute_constraint_penalty(company, filters):
    """
    Return a multiplier between 0 and 1 based on how well the company
    satisfies the structured filters.

        1.0  = passes all filters
        0.05 = hard fail on a known field (e.g. wrong country)
        0.3–0.6 = data is missing, partial benefit of the doubt
    """
    penalty = 1.0
    country = get_country_code(company)

    # --- geography ---
    if "geo" in filters:
        allowed_countries = filters["geo"]
        if country and country not in allowed_countries:
            penalty *= 0.05     # wrong country → almost eliminated
        elif not country:
            penalty *= 0.3      # unknown country → reduced but not killed

    # --- public status ---
    if filters.get("is_public") and not company.get("is_public"):
        penalty *= 0.05

    # --- employee count ---
    employee_count = company.get("employee_count")

    if "min_emp" in filters:
        minimum_employees = filters["min_emp"]
        if employee_count and employee_count < minimum_employees:
            penalty *= 0.05
        elif employee_count is None:
            penalty *= 0.4

    if "max_emp" in filters:
        maximum_employees = filters["max_emp"]
        if employee_count and employee_count > maximum_employees:
            penalty *= 0.05
        elif employee_count is None:
            penalty *= 0.6

    # --- revenue ---
    if "min_rev" in filters:
        minimum_revenue = filters["min_rev"]
        company_revenue = company.get("revenue")
        if company_revenue and company_revenue < minimum_revenue:
            penalty *= 0.05
        elif company_revenue is None:
            penalty *= 0.4

    # --- founding year ---
    if "min_year" in filters:
        minimum_year = filters["min_year"]
        year_founded = company.get("year_founded")
        if year_founded and year_founded <= minimum_year:
            penalty *= 0.05
        elif year_founded is None:
            penalty *= 0.5

    # --- business model ---
    business_model_text = " ".join(company.get("business_model") or []).lower()

    if filters.get("b2b"):
        if "b2b" not in business_model_text and "business-to-business" not in business_model_text:
            penalty *= 0.3

    if filters.get("saas"):
        if "saas" not in business_model_text and "software" not in business_model_text:
            penalty *= 0.3

    return penalty


# --- Main ranking ---

def rank_companies(query, expanded_query, companies, document_tokens,
                   inverse_doc_freq, filters, top_n=10):
    """
    Score and rank companies using 4 weighted signals × constraint penalties.

    Signals:
        1. TF-IDF text similarity (on synonym-expanded query)
        2. NAICS industry code match
        3. Core offerings keyword overlap
        4. Supply chain role relevance

    Returns: list of (score, company) tuples, sorted highest first.
    """
    text_scores = compute_text_similarity(expanded_query, document_tokens,
                                          inverse_doc_freq)
    naics_prefixes = filters.get("naics_prefixes", [])
    required_role  = filters.get("role")

    scored_companies = []

    for index, company in enumerate(companies):
        # compute each signal
        text_signal     = text_scores[index]
        naics_signal    = compute_naics_match(company, naics_prefixes)
        offerings_signal = compute_offerings_overlap(company, expanded_query)
        role_signal     = compute_supplier_role_score(company, required_role) if required_role else 0.0

        # weighted combination
        raw_score = (TEXT_SIMILARITY_WEIGHT  * text_signal
                   + NAICS_MATCH_WEIGHT     * naics_signal
                   + OFFERINGS_MATCH_WEIGHT * offerings_signal
                   + SUPPLY_ROLE_WEIGHT     * role_signal)

        # apply constraint penalties
        penalty = compute_constraint_penalty(company, filters)
        final_score = raw_score * penalty

        if final_score > 0.001:
            scored_companies.append((final_score, company))

    scored_companies.sort(key=lambda entry: entry[0], reverse=True)
    return scored_companies[:top_n]
