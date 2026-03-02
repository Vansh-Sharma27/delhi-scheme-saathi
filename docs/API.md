# API Documentation

Base URL: `http://localhost:8000`

Interactive documentation available at `/docs` (Swagger UI).

## Health Check

### GET /health

Returns service health status and database connectivity.

**Response**
```json
{
  "status": "ok",
  "database": "connected",
  "schemes_count": 5
}
```

| Field | Description |
|-------|-------------|
| status | `ok` or `degraded` |
| database | `connected`, `disconnected`, or error message |
| schemes_count | Number of active schemes in database |

---

## Schemes

### GET /api/schemes

List all schemes, optionally filtered by life event.

**Query Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| life_event | string | Filter by life event (e.g., HOUSING, DEATH_IN_FAMILY) |
| limit | int | Maximum results (default: 10) |

**Example**
```bash
curl "http://localhost:8000/api/schemes?life_event=HOUSING&limit=5"
```

**Response**
```json
{
  "schemes": [
    {
      "id": "SCH-DELHI-001",
      "name": "Pradhan Mantri Awas Yojana - Urban 2.0",
      "name_hindi": "प्रधानमंत्री आवास योजना - शहरी 2.0",
      "department": "Ministry of Housing and Urban Affairs",
      "level": "central",
      "benefits_amount": 250000,
      "benefits_frequency": "as-needed",
      "eligibility": {
        "min_age": 18,
        "max_income": 1800000,
        "categories": ["EWS", "LIG", "MIG-I", "MIG-II"]
      },
      "life_events": ["HOUSING"],
      "is_active": true
    }
  ],
  "total": 1,
  "life_event": "HOUSING"
}
```

### GET /api/scheme/{scheme_id}

Get full details for a specific scheme including documents and rejection rules.

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| scheme_id | string | Scheme identifier (e.g., SCH-DELHI-001) |

**Example**
```bash
curl "http://localhost:8000/api/scheme/SCH-DELHI-003"
```

**Response**
```json
{
  "scheme": {
    "id": "SCH-DELHI-003",
    "name": "Delhi Pension Scheme to Women in Distress (Widow Pension)",
    "name_hindi": "कठिनाई में महिलाओं के लिए दिल्ली पेंशन योजना (विधवा पेंशन)",
    "benefits_amount": 2500,
    "benefits_frequency": "monthly",
    "eligibility": {
      "min_age": 18,
      "max_income": 100000,
      "genders": ["female"],
      "domicile_required": true
    },
    "documents_required": ["DOC-AADHAAR", "DOC-DEATH-CERT", "DOC-INCOME-CERT"],
    "helpline": {
      "phone": "+91-11-23070378, +91-11-23380567",
      "website": "https://wcd.delhi.gov.in"
    }
  },
  "documents": [
    {
      "id": "DOC-AADHAAR",
      "name": "Aadhaar Card",
      "issuing_authority": "UIDAI",
      "fee": "Free"
    }
  ],
  "rejection_rules": [
    {
      "id": "REJ-003-001",
      "rule_type": "age",
      "description": "Applicant not meeting minimum age requirement",
      "severity": "critical",
      "prevention_tip": "Verify age from birth certificate matches Aadhaar"
    }
  ]
}
```

---

## Documents

### GET /api/document/{document_id}

Get document details with prerequisites and issuing offices.

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| document_id | string | Document identifier (e.g., DOC-INCOME-CERT) |

**Example**
```bash
curl "http://localhost:8000/api/document/DOC-INCOME-CERT"
```

**Response**
```json
{
  "document": {
    "id": "DOC-INCOME-CERT",
    "name": "Income Certificate",
    "name_hindi": "आय प्रमाण पत्र",
    "issuing_authority": "Sub-Divisional Magistrate (SDM)",
    "online_portal": "https://edistrict.delhigovt.nic.in",
    "fee": "10",
    "fee_bpl": "Free",
    "processing_time": "7-15 days",
    "prerequisites": ["DOC-AADHAAR", "DOC-RATION-CARD"],
    "common_mistakes": [
      "Name mismatch between documents",
      "Incomplete address proof"
    ]
  },
  "prerequisites": [
    {
      "id": "DOC-AADHAAR",
      "name": "Aadhaar Card",
      "fee": "Free"
    }
  ],
  "offices": [
    {
      "id": "OFF-SDM-CENTRAL",
      "name": "SDM Office - Central Delhi",
      "address": "Minto Road, New Delhi",
      "working_hours": "Mon-Sat 9:30 AM - 5:00 PM"
    }
  ]
}
```

---

## Offices

### GET /api/csc/nearest

Find nearest government offices by location or district.

**Query Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| lat | float | Latitude (required if no district) |
| lng | float | Longitude (required if no district) |
| district | string | District name (required if no lat/lng) |
| office_type | string | Filter by type (optional) |
| limit | int | Maximum results (default: 5) |

**Example (by location)**
```bash
curl "http://localhost:8000/api/csc/nearest?lat=28.6139&lng=77.2090&limit=3"
```

**Example (by district)**
```bash
curl "http://localhost:8000/api/csc/nearest?district=Central"
```

**Response**
```json
{
  "offices": [
    {
      "id": "OFF-SDM-CENTRAL",
      "name": "SDM Office - Central Delhi",
      "type": "SDM Office",
      "address": "Minto Road, New Delhi - 110002",
      "district": "Central",
      "latitude": 28.6298,
      "longitude": 77.2311,
      "phone": "011-23237655",
      "working_hours": "Mon-Sat 9:30 AM - 5:00 PM",
      "services": ["DOC-INCOME-CERT", "DOC-DOMICILE", "DOC-CASTE-CERT"],
      "distance_km": 2.87
    }
  ],
  "total": 1,
  "query_type": "location",
  "query_location": [28.6139, 77.209]
}
```

---

## Life Events

### GET /api/life-events

List all supported life event categories.

**Example**
```bash
curl "http://localhost:8000/api/life-events"
```

**Response**
```json
{
  "life_events": [
    {
      "key": "HOUSING",
      "display_name": "Housing & Property",
      "display_name_hindi": "आवास एवं संपत्ति",
      "aliases": ["buying_home", "constructing_home", "renting_home"]
    },
    {
      "key": "DEATH_IN_FAMILY",
      "display_name": "Death in Family",
      "display_name_hindi": "परिवार में मृत्यु",
      "aliases": ["death_of_spouse", "widowhood", "bereavement"]
    }
  ]
}
```

---

## Chat

### POST /api/chat

Direct chat endpoint for testing (bypasses Telegram).

**Request Body**
```json
{
  "user_id": "unique_user_identifier",
  "message": "User message in Hindi/English/Hinglish"
}
```

**Example**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test123", "message": "Namaste, mujhe pension chahiye"}'
```

**Response**
```json
{
  "response": "नमस्ते! मैं दिल्ली स्कीम साथी हूँ। आप मुझे बताएं, आज आपको किस तरह की सहायता चाहिए?",
  "next_state": "UNDERSTANDING",
  "schemes": [],
  "documents": [],
  "rejection_warnings": []
}
```

---

## Telegram Webhook

### POST /webhook/telegram

Webhook endpoint for Telegram Bot API.

**Request Body**

Standard Telegram Update object. See [Telegram Bot API documentation](https://core.telegram.org/bots/api#update).

**Response**
```json
{
  "status": "ok"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Provide either lat+lng or district parameter"
}
```

### 404 Not Found
```json
{
  "detail": "Scheme SCH-INVALID not found"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Database connection not available"
}
```

---

## Rate Limits

No rate limiting is currently implemented. For production deployment, consider adding rate limiting at the API Gateway level.

## Authentication

No authentication is required for API endpoints. For production deployment, consider adding API key authentication for non-Telegram endpoints.
