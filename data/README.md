# Delhi Scheme Saathi - Scheme Data

Production-ready seed data for Delhi Scheme Saathi, covering 5 government welfare schemes for Delhi residents.

## Schemes Included

| ID | Scheme | Department | Level | Benefits |
|----|--------|------------|-------|----------|
| SCH-DELHI-001 | PMAY-U 2.0 (Housing for All) | MoHUA | Central | Up to Rs 2.5 lakh |
| SCH-DELHI-003 | Widow Pension Scheme | WCD, GNCTD | State | Rs 2,500/month |
| SCH-DELHI-005 | RGSRY (Self-Employment) | DKVIB, GNCTD | State | Up to Rs 3 lakh loan |
| SCH-DELHI-006 | ELSD (Education Loan) | DSFDC, GNCTD | State | Up to Rs 7.5 lakh |
| SCH-DELHI-007 | Delhi Arogya Kosh (Health) | Health Dept, GNCTD | State | Up to Rs 5 lakh |

## File Structure

```
data/
├── Individual Scheme Files (with related docs & rules)
│   ├── pmay_u_2.0.json          # PMAY-U 2.0 Housing scheme
│   ├── widow_pension.json        # Delhi Widow Pension
│   ├── rgsry.json                # RGSRY Self-Employment loan
│   ├── elsd.json                 # Education Loan Scheme
│   └── delhi_arogya_kosh.json    # Delhi Arogya Kosh (Health)
│
├── Consolidated Files
│   ├── all_schemes.json          # All 5 schemes
│   ├── all_documents.json        # 29 documents
│   ├── all_offices.json          # 16 offices (CSC, SDM, Depts)
│   ├── all_rejection_rules.json  # 46 rejection rules
│   └── life_events_taxonomy.json # Life event categories
```

## Data Verification

All scheme data has been verified against official government sources:

- **PMAY-U 2.0**: [pmay-urban.gov.in](https://pmay-urban.gov.in/)
- **Widow Pension**: [wcd.delhi.gov.in](https://wcd.delhi.gov.in/)
- **RGSRY**: [dkvib.delhi.gov.in](https://dkvib.delhi.gov.in/) - Verified against official salient features document
- **ELSD**: [dsfdc.delhi.gov.in](https://dsfdc.delhi.gov.in/)
- **DAK**: [health.delhi.gov.in](https://health.delhi.gov.in/health/delhi-arogya-kosh)

Last verified: 2026-02-28

## Schema Overview

Each scheme file contains:

```json
{
  "scheme": {
    "id": "SCH-DELHI-XXX",
    "name": "Scheme Name",
    "name_hindi": "योजना का नाम",
    "department": "...",
    "level": "central|state",
    "description": "...",
    "benefits_summary": "...",
    "benefits_amount": 250000,
    "eligibility": {
      "min_age": 18,
      "max_age": 50,
      "max_income": 300000,
      "categories": ["SC", "ST", "OBC", ...],
      "domicile_states": ["delhi"]
    },
    "documents_required": ["DOC-AADHAAR", ...],
    "life_events": ["HOUSING", "HEALTH_CRISIS", ...],
    "helpline": { "phone": [...], "address": "..." },
    "metadata": { ... }
  },
  "documents": [...],
  "rejection_rules": [...]
}
```

## Life Events Taxonomy

Schemes are categorized by life events using UPPERCASE keys:

| Key | Description | Example Schemes |
|-----|-------------|-----------------|
| HOUSING | Home purchase/construction | PMAY-U 2.0 |
| HEALTH_CRISIS | Medical emergencies | Delhi Arogya Kosh |
| DEATH_IN_FAMILY | Bereavement, widowhood | Widow Pension |
| MARITAL_DISTRESS | Divorce, separation | Widow Pension |
| BUSINESS_STARTUP | Self-employment | RGSRY |
| EDUCATION | Higher education | ELSD |
| JOB_LOSS | Unemployment | RGSRY |

## Generating Embeddings

Embeddings are not included in the JSON files. Generate them using Voyage AI:

```python
import voyageai

vo = voyageai.Client(api_key="YOUR_API_KEY")

# For each scheme
text = f"{scheme['name']} {scheme['description']} {scheme['benefits_summary']}"
result = vo.embed(texts=[text], model="voyage-multilingual-2", input_type="document")
scheme['description_embedding'] = result.embeddings[0]  # 1024 dimensions
```

## Database Seeding

See `/database/seed/schema.sql` for PostgreSQL table definitions with pgvector support.

```bash
# Create database
psql -f schema.sql

# Seed data using provided scripts or direct JSON import
```

## Key Data Points (Verified)

| Scheme | Benefit Amount | Income Limit | Age |
|--------|----------------|--------------|-----|
| PMAY-U 2.0 | Rs 2.5L (EWS) | Rs 3L/6L/9L | Any |
| DAK | Up to Rs 5L | Rs 3L | Any |
| Widow Pension | Rs 2,500/month | Rs 1L | 18+ |
| RGSRY | Rs 3L loan + 15% subsidy | None | 18-50 |
| ELSD | Rs 7.5L (India) / Rs 15L (Abroad) | Rs 5L | Any |
