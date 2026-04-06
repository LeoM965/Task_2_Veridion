import json
from data_loader import load_companies, build_company_text, get_country_code
from query_parser import parse_query, expand_query_with_synonyms
from search import build_tfidf_index, rank_companies

QUERIES = [
    "Logistic companies in Romania",
    "Public software companies with more than 1,000 employees.",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Construction companies in the United States with revenue over $50 million",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Fast-growing fintech companies competing with traditional banks in Europe.",
    "E-commerce companies using Shopify or similar platforms",
    "Renewable energy equipment manufacturers in Scandinavia",
    "Companies that manufacture or supply critical components for electric vehicle battery production",
]


def print_results(query, results):
    """Display ranked results for a single query."""
    print("=" * 70)
    print(f"  {query}")
    print("=" * 70)

    if not results:
        print("  (no matches found)\n")
        return

    for position, (score, company) in enumerate(results, 1):
        name        = company.get("operational_name", "?")
        website     = company.get("website", "-")
        country     = get_country_code(company).upper() or "??"
        description = (company.get("description") or "")[:100]
        print(f"  {position:>2}. [{score:.4f}]  {name}  ({website}, {country})")
        print(f"      {description}...")
    print()


def save_results_to_json(all_query_results, output_path="results.json"):
    """Save all query results to a JSON file."""
    output = []

    for query, results in all_query_results:
        matches = []
        for score, company in results:
            matches.append({
                "score":   round(score, 4),
                "name":    company.get("operational_name", "?"),
                "website": company.get("website", "-"),
                "country": get_country_code(company).upper() or "??",
            })
        output.append({"query": query, "matches": matches})

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


def main():
    # 1. load company data
    companies = load_companies("companies.jsonl")
    print(f"Loaded {len(companies)} companies.\n")

    # 2. build the TF-IDF search index
    company_texts = [build_company_text(company) for company in companies]
    document_tokens, inverse_doc_freq = build_tfidf_index(company_texts)
    print(f"Index ready: {len(inverse_doc_freq)} terms.\n")

    # 3. run every query
    all_query_results = []

    for query in QUERIES:
        filters        = parse_query(query)
        expanded_query = expand_query_with_synonyms(query)
        results        = rank_companies(query, expanded_query, companies,
                                        document_tokens, inverse_doc_freq, filters)
        print_results(query, results)
        all_query_results.append((query, results))

    # 4. export results
    save_results_to_json(all_query_results)


if __name__ == "__main__":
    main()
