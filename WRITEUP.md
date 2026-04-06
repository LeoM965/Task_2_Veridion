# Writeup — Company Qualification System

## 1. Approach

### Architecture

The system uses a **3-stage pipeline**: parse → score → filter.

```
User Query
    │
    ▼
┌──────────────┐
│ Query Parser │ ── extracts structured filters (geo, employees, revenue, etc.)
│              │ ── expands query with synonyms ("logistics" → + freight, shipping, ...)
│              │ ── detects NAICS industry prefixes
│              │ ── detects supply chain role (supplier vs buyer)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Multi-Signal │ ── Signal 1: TF-IDF cosine similarity (50%)
│   Scorer     │ ── Signal 2: NAICS industry code match (20%)
│              │ ── Signal 3: Core offerings keyword overlap (15%)
│              │ ── Signal 4: Supply chain role relevance (15%)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Constraint  │ ── applies soft penalties for filter violations
│  Penalties   │ ── hard fail (×0.05) vs missing data (×0.3–0.5)
└──────┬───────┘
       │
       ▼
   Ranked Results
```

### File structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads JSONL, normalizes messy fields, extracts NAICS codes/labels |
| `query_parser.py` | Regex-based filter extraction + synonym expansion + NAICS hints |
| `search.py` | TF-IDF engine + 4 scoring signals + constraint penalties |
| `solution.py` | Entry point runs 12 queries, prints results, exports JSON |

### Key design decisions

**Synonym expansion.** The query "logistics companies" is expanded with related terms like "freight", "shipping", "warehousing", "supply chain", "forwarding". This dramatically improves recall without any external API calls.

**NAICS code matching.** Each query keyword maps to relevant NAICS code prefixes. A company classified under NAICS 488 (Freight Transportation) gets a direct boost when the query mentions "logistics". This catches companies even when their description doesn't use the exact query words.

**Supply chain role detection.** When a query asks for "suppliers of X", the system checks if a company's description and offerings contain supplier signals ("manufacturer", "supplier", "wholesale", "component"), not just the product keyword. This helps distinguish packaging suppliers from cosmetics brands.

**Soft penalties instead of hard filters.** A company with unknown location gets `score × 0.3` instead of being dropped. This keeps potentially relevant companies visible while ranking them below confirmed matches.

**Multi-signal weighted scoring.** Final score = `(0.50 × text_sim + 0.20 × naics_match + 0.15 × offerings_overlap + 0.15 × role_relevance) × penalty`. Each signal captures a different dimension of relevance.

### Why this design?

Zero external dependencies. The entire system runs on Python's standard library in under a second. No API keys, no pip installs, no Docker. This makes it portable and reproducible on any machine.

---

## 2. Tradeoffs

| Optimized for | At the expense of |
|---|---|
| Speed (< 100ms per query) | No deep semantic reasoning |
| Zero dependencies | Can't use sentence-transformers or spaCy |
| Missing data tolerance | Some false positives survive with lower scores |
| Simple readable code | Less sophisticated than an LLM pipeline |
| Synonym expansion | Manually curated synonym lists, not learned |

The biggest tradeoff is **accuracy on vague queries**. "Fast-growing fintech companies" requires growth data which doesn't exist in the dataset. I can only rank on text similarity to "fintech" + location filter. A real system would need time-series financials.

The synonym lists are **hand-curated** rather than learned from a corpus. This means they work well for the 12 test queries but would need extension for arbitrary domains. In production, I would replace these with a word2vec or WordNet-based expansion.

---

## 3. Error Analysis

### Where it works well

- **Structured queries** like "Public software companies with more than 1,000 employees" — the constraint parser catches every field, TF-IDF handles "software", NAICS codes provide industry backup.
- **Location-based queries** — country codes in the data are reliable, and the region mappings (Europe, Scandinavia) cover multi-country queries.
- **Industry queries** — NAICS code matching catches companies even when their description uses different terminology than the query.

### Where it struggles

**"Companies that could supply packaging materials for cosmetics"**
TF-IDF matches on both "packaging" and "cosmetics" words. A cosmetics brand that mentions "packaging" in its description can rank alongside actual packaging suppliers. The supply chain role detection helps, but a company saying "we package our products beautifully" still triggers supplier signals falsely.

**"E-commerce companies using Shopify"**
If "Shopify" doesn't appear in a company's description, the system can't find it. The dataset doesn't contain tech stack information. This is a data gap, not a system design flaw.

**Synonym gaps**
A company saying "motor vehicles" won't match a query about "cars" unless that specific synonym mapping exists. TF-IDF only matches exact tokens, and synonym lists are finite.

**"Fast-growing" → no growth signal**
The dataset has a snapshot of revenue and employees, but no historical data. I can't infer growth rate. The system falls back to text matching on "fintech" + geographic filters, which is an incomplete answer.

---

## 4. Scaling

For 100,000+ companies per query:

1. **Database pre-filtering.** Index `country_code`, `is_public`, `employee_count`, `revenue`, `year_founded` in PostgreSQL or SQLite. Run constraint filters first to cut 100K → 5K candidates before any text matching.

2. **Replace TF-IDF with Elasticsearch/BM25.** Elasticsearch handles millions of documents natively, supports geographic queries, and BM25 is a proven improvement over raw TF-IDF.

3. **Pre-computed embeddings for complex queries.** Store sentence embeddings (e.g., `all-MiniLM-L6-v2`) in a FAISS index. Use vector search only for queries where the regex parser finds no structured constraints  the vague/interpretive ones.

4. **LLM verification on top-N only.** After the cheap pipeline narrows 100K → 20 candidates, send those 20 to GPT-4 for final yes/no judgment. Cost: ~$0.01 per query instead of $5 for full-set LLM calls.

5. **Async batch processing.** Use Python `asyncio` with `aiohttp` to parallelize LLM calls on the top-N set, turning 20 sequential calls into 3–4 batched rounds.

---

## 5. Failure Modes

### Confident but incorrect results

- A **logistics software** company ranking #1 for "logistics companies" it mentions logistics everywhere in its description and NAICS label, but doesn't actually do physical logistics. The system sees strong text + NAICS signals and rates it highly.

- A **cosmetics brand** ranking high for "packaging suppliers for cosmetics"  similarity ≠ supply chain relationship. The company mentions "packaging" in its description because it *uses* packaging, not because it *supplies* it.

- A company **near a border** ranking for the wrong country  if the address says "Germany" but the company primarily operates in Poland, the system trusts the registered address blindly.

### What I'd monitor in production

| Metric | Why |
|--------|-----|
| **Empty result rate** | A reasonable query returning 0 results means the parser or scorer has a bug |
| **Score distribution** | If the top result scores 0.02 and #2 scores 0.01, confidence is low  the system should say "no confident matches" |
| **Score gap ratio** | A large gap between #1 and #2 suggests high-confidence match; a flat distribution suggests ambiguity |
| **Filter hit rate** | Track how often each filter type is triggered zero hits on a filter means the regex might be broken |
| **User click-through** | In production, measure which ranked results users actually click to calibrate scoring weights |
| **Synonym coverage** | Log queries where TF-IDF returns zero hits these likely need new synonym entries |
