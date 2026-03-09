# Delhi Scheme Saathi - Requirements Document

## Project Overview

**Project Name:** Delhi Scheme Saathi  
**Tagline:** "Hum sirf scheme nahi batate, apply karwa ke dikhate hain"  
**Problem Statement:** Build an AI-powered solution that improves access to information, resources, or opportunities for communities and public systems.

### Core Problem Analysis

The real challenge is not awareness but execution:
1. Complex document requirements with no guidance on WHERE to get documents
2. Applications rejected for minor errors (name mismatch, wrong authority, expired certificates)  
3. No human help when stuck in the process
4. Existing solutions (MyScheme, Jugalbandi) stop at awareness - nobody helps with execution

### Solution Vision

A voice-first Telegram chatbot that goes beyond awareness to provide complete execution assistance for Delhi government schemes. Built on AWS Mumbai region infrastructure with Sarvam AI (primary) and Bhashini/AI4Bharat (fallback) for Hindi voice processing, AWS Bedrock Nova (primary) and xAI Grok (fallback) for conversational intelligence, and Jina AI + Voyage AI for multilingual embeddings.

**Core Principle:** LLM handles HOW we talk (natural conversation, intent understanding, response generation). Rules engine handles WHAT we say about eligibility (structured filtering, document requirements, rejection rules). This separation ensures conversational fluency without hallucinating scheme-specific facts.

## Target Users

### Primary Users
- **Delhi domicile holders** (voter ID / ration card holders)
- **Semi-literate or vernacular-first users** comfortable with voice interaction
- **People facing life events:** job loss, childbirth, disability, education, marriage, business startup

### User Personas
1. **Priya (28, New Mother):** Recently gave birth, needs maternity benefits, comfortable with Hindi voice messages
2. **Rajesh (45, Unemployed):** Lost job due to company closure, needs unemployment benefits, limited English
3. **Sunita (58, Widow):** Lost husband, needs widow pension, rejected due to wrong certificate authority

## Functional Requirements

### FR1: Voice-First Hindi Interface

**User Story:** As a Hindi-speaking Delhi resident, I want to interact with the system using voice messages so that I can get help without typing.

**Acceptance Criteria:**
- AC1.1: System accepts voice messages via Telegram Bot API
- AC1.2: Voice transcription accuracy > 85% for conversational Hindi
- AC1.3: End-to-end voice response latency < 8 seconds
- AC1.4: Supports Hindi, English, and Hinglish with automatic language detection
- AC1.5: Fallback to text mode if voice processing fails (low-confidence threshold at 0.5)
- AC1.6: Primary voice pipeline: Sarvam AI (Saaras v3 STT / Bulbul v3 TTS), Fallback: Bhashini (AI4Bharat IndicWhisper / IndicTTS)
- AC1.7: Multi-language STT probing — tries multiple language candidates and picks the highest-quality transcript

**Priority:** High
**Dependencies:** Sarvam AI API integration (primary), Bhashini API integration (fallback)

### FR2: Life-Event Based Discovery

**User Story:** As a user facing a life situation, I want to describe my situation naturally so that the system can identify relevant schemes without me knowing scheme names.

**Acceptance Criteria:**
- AC2.1: Correctly identify life event from natural language with >90% accuracy
- AC2.2: Map each life event to 5-15 relevant schemes
- AC2.3: Handle ambiguous queries with clarifying questions
- AC2.4: Support life event categories:
  - HOUSING, MARRIAGE, CHILDBIRTH, EDUCATION
  - HEALTH_CRISIS, DEATH_IN_FAMILY, MARITAL_DISTRESS
  - JOB_LOSS, BUSINESS_STARTUP, WOMEN_EMPOWERMENT

**Example Mappings:**
- "Mera accident ho gaya" → Health Crisis schemes
- "Ghar mein baccha aaya" → Childbirth schemes
- "Beti ki shaadi hai" → Marriage assistance
- "Mujhe ghar chahiye" → Housing schemes
- "Meri naukri chali gayi" → Job Loss schemes

**Priority:** High  
**Dependencies:** NLP model training, scheme database

### FR3: Conversational Eligibility Profiling

**User Story:** As a user, I want to provide my details through natural conversation so that I don't have to fill complex forms.

**Acceptance Criteria:**
- AC3.1: Collect minimum required attributes in 3-4 conversational turns
- AC3.2: Don't re-ask attributes already provided in session
- AC3.3: Handle corrections gracefully ("Actually, my age is 35, not 30")
- AC3.4: Required attributes (scheme-aware, varies by life event): age, gender, annual_income, caste_category
- AC3.5: Conditional attributes: employment_status, marital_status, has_bpl_card
- AC3.6: Optional attributes: disability_percentage, district, latitude/longitude
- AC3.7: Dual extraction pipeline: LLM-based entity extraction merged with regex pattern extraction (rule-based takes precedence)
- AC3.8: Contextual extraction: bare numeric responses interpreted based on `currently_asking` field (e.g., bare "19" → age if bot just asked for age)

**Priority:** High  
**Dependencies:** Conversation state management

### FR4: Hybrid Scheme Retrieval

**User Story:** As a user, I want to get the most relevant schemes for my situation so that I don't waste time on irrelevant options.

**Acceptance Criteria:**
- AC4.1: Three-stage matching: life event filtering + structured eligibility filtering + semantic ranking
- AC4.2: Stage 1 SQL filtering by life event category
- AC4.3: Stage 2 SQL filtering by age, income, category with post-filter for non-SQL restrictions (EWS/LIG/MIG income bands)
- AC4.4: Stage 3 semantic ranking using Jina AI (primary) / Voyage AI (fallback) embeddings (1024-dim, pgvector HNSW index)
- AC4.5: Optional LLM-based relevance judging when deterministic scores are ambiguous (score < 0.85 or top-two gap < 0.15)
- AC4.6: Return top 5 schemes ranked by composite relevance score
- AC4.5: Retrieval latency < 500ms
- AC4.6: Precision > 80% (returned schemes are relevant)
- AC4.7: Recall > 70% (relevant schemes are returned)

**Priority:** High  
**Dependencies:** Vector database, embedding model

### FR5: Document Procurement Guide (KEY DIFFERENTIATOR)

**User Story:** As a user, I want to know exactly where and how to get each required document so that I can complete my application without confusion.

**Acceptance Criteria:**
- AC5.1: For each required document, provide complete procurement guidance
- AC5.2: Document info includes:
  - Issuing authority and alternate authorities
  - Prerequisites needed to get the document
  - Fee structure (regular and BPL rates)
  - Processing time and validity period
  - Format requirements and common mistakes
- AC5.3: Delhi office database with 16+ government offices (MVP)
- AC5.4: Haversine distance calculation from user's approximate location
- AC5.5: Online portal links verified and working
- AC5.6: Document info available for 29 common government documents covering all 5 MVP schemes

**Example Document Info:**
```
Income Certificate (आय प्रमाण पत्र)
- Issuing Authority: Sub-Divisional Magistrate (SDM)
- Online Option: edistrict.delhigovt.nic.in
- Prerequisites: Aadhaar, Ration Card, Self-declaration
- Fee: ₹50 (Free for BPL)
- Processing Time: 7-15 working days
- Validity: 6 months from issue date
- Common Mistake: Name must match Aadhaar exactly
```

**Priority:** High (Key Differentiator)  
**Dependencies:** Government office database, document database

### FR6: Rejection Prevention System (KEY DIFFERENTIATOR)

**User Story:** As a user, I want to know common rejection reasons before applying so that I can avoid mistakes and increase my chances of approval.

**Acceptance Criteria:**
- AC6.1: 5-10 rejection rules documented per major scheme
- AC6.2: Prevention tips are actionable and specific
- AC6.3: Warnings shown BEFORE user starts application
- AC6.4: Rejection categories covered:
  - NAME_MISMATCH: Name spelling differs across documents
  - WRONG_AUTHORITY: Certificate from incorrect issuing authority
  - EXPIRED_DOCUMENT: Document validity expired
  - WRONG_CATEGORY: Incorrect SC/ST/OBC/General selection
  - TIMELINE_VIOLATION: Applied after deadline
  - FORMAT_ERROR: Wrong file format, size, or missing attestation

**Priority:** High (Key Differentiator)  
**Dependencies:** Rejection rules database

### FR7: Application Walkthrough

**User Story:** As a user, I want step-by-step guidance for submitting my application so that I can complete the process successfully.

**Acceptance Criteria:**
- AC7.1: Walkthrough available for 5 schemes (MVP), expanding to 25+
- AC7.2: Instructions verified against actual portal
- AC7.3: Both online and offline options provided
- AC7.4: Content includes:
  - Online application URL
  - Step-by-step instructions
  - Document upload specifications (format, size)
  - Offline alternative (which office, what to carry)
  - Processing time estimate
  - Status tracking method
  - Helpline numbers

**Priority:** Medium  
**Dependencies:** Scheme application processes

### FR8: CSC Locator and Human Handoff

**User Story:** As a user stuck in the process, I want to find the nearest Common Service Centre and get human help so that I can complete my application.

**Acceptance Criteria:**
- AC8.1: 500+ Delhi CSCs in database
- AC8.2: Distance calculation within 500m accuracy
- AC8.3: Contact numbers verified and working
- AC8.4: CSC information includes:
  - Address, contact, working hours
  - Services offered and fee structure
  - Operator name (if available)
- AC8.5: Generate handoff summary including: user profile, eligible schemes, documents needed

**Priority:** Medium  
**Dependencies:** CSC database, location services

### FR9: Session Management

**User Story:** As a user, I want the system to remember our conversation context so that I don't have to repeat information.

**Acceptance Criteria:**
- AC9.1: Session persists for 7 days from last message via DynamoDB TTL (target users may take days to gather documents between interactions)
- AC9.2: Context maintained across 20+ conversational turns via 12-message sliding window (6 completed turns) plus LLM-generated working memory
- AC9.3: Working memory system: deterministic profile facts + LLM-generated conversation summary, refreshed every 8 turns or when context exceeds ~6000 tokens
- AC9.4: Background memory refresh via SQS queue (async) to avoid blocking user responses
- AC9.5: Session includes:
  - User profile (collected attributes with scheme-aware required fields)
  - Message history (last 12 messages as sliding window)
  - Working memory (LLM-generated summary + deterministic profile facts + active scheme IDs + pending action)
  - Conversation state (10-state FSM)
  - Discussed schemes (avoid repetition)
  - Selected scheme ID (for deep-dive navigation)
  - Presented schemes (for inline keyboard recall)
  - Language preference (auto-detected or user-locked)
  - Currently asking field (for contextual bare-value extraction)
  - Skipped fields (profile fields user declined to answer)
  - Awaiting profile change guard (prevents re-matching loops)
  - Completed turn count and memory refresh tracking

**Conversation States (10-state FSM):**
1. GREETING - Initial welcome and language detection
2. SITUATION_UNDERSTANDING - User describes life event
3. PROFILE_COLLECTION - Gathering eligibility attributes (scheme-aware)
4. SCHEME_MATCHING - Running 3-stage hybrid retrieval (transient — auto-transitions)
5. SCHEME_PRESENTATION - Showing eligible schemes with inline keyboard
6. SCHEME_DETAILS - Deep dive into one scheme (benefits, eligibility match explanation)
7. DOCUMENT_GUIDANCE - Document procurement with prerequisites, fees, offices
8. REJECTION_WARNINGS - Common mistakes and prevention tips
9. APPLICATION_HELP - Step-by-step online/offline guidance
10. CSC_HANDOFF - Connecting to human help at nearest CSC

**State Transition Notes:**
- Backward transitions are supported: users can say "go back to documents" from REJECTION_WARNINGS or "show me the scheme list again" from APPLICATION_HELP. The state machine does not force a strictly linear path.
- CSC_HANDOFF is not terminal: users who return after visiting a CSC can re-enter at SCHEME_PRESENTATION or GREETING.

**Priority:** High  
**Dependencies:** Session storage system

## Non-Functional Requirements

### NFR1: Performance
- Voice end-to-end response: < 5 seconds
- Text response: < 3 seconds  
- Scheme retrieval: < 500ms
- Concurrent sessions: 100 (MVP), 1000 (production)

### NFR2: Security
- TLS 1.3 for all connections
- Session data expires via DynamoDB TTL (7-day default, configurable)
- Rate limiting: via API Gateway throttling
- Input sanitization for prompt injection prevention

### NFR3: Availability
- 99.5% uptime target
- Graceful degradation if Sarvam AI unavailable (Bhashini fallback, then text-only mode)
- LLM fallback chain: AWS Bedrock Nova 2 Lite → xAI Grok → safe error message
- Embedding fallback chain: Jina AI → Voyage AI → skip vector ranking

### NFR4: Scalability
- Serverless auto-scaling architecture
- Database read replicas for high load
- CDN for static assets and audio files

### NFR5: Data Sovereignty
- All data stored in AWS Mumbai region (ap-south-1)
- Voice processing through Sarvam AI (India-based) with Bhashini (AI4Bharat, IIT Madras) fallback — both India-hosted
- LLM processing via AWS Bedrock (India region) when available, xAI Grok as fallback
- All language models accessed through India-region endpoints where available

### NFR6: Usability
- Voice transcription accuracy > 85% for conversational Hindi
- Automatic language detection before transcription (Hindi, English, Hinglish) — the system does not assume Hindi
- Transcription confidence scoring: low-confidence results (< 0.6) trigger a text fallback prompt instead of acting on potentially misheard input
- Support for code-mixed Hindi-English (Hinglish)
- Graceful handling of background noise and accents

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Schemes discovered per user | ≥ 3 | Analytics tracking |
| Document guidance helpfulness | ≥ 80% positive | User feedback |
| Application attempt rate | ≥ 40% | Follow-up surveys |
| Successful benefit receipt | ≥ 20% (vs ~8% baseline) | 3-month follow-up |
| User return rate (30 days) | ≥ 30% | User analytics |
| Net Promoter Score (NPS) | ≥ 40 | User surveys |

## Constraints and Assumptions

### Technical Constraints
- Must use Telegram Bot API (primary channel for MVP; WhatsApp planned for future)
- Must integrate with Sarvam AI (primary) and Bhashini (fallback) for Hindi voice processing
- Must comply with Indian data localization requirements
- AWS Bedrock availability varies by region — xAI Grok as universal fallback

### Business Constraints
- Focus on Delhi schemes only (MVP scope)
- 5 schemes deep (PMAY-U Housing, Delhi Arogya Kosh Health, Widow Pension, RGSRY Self-Employment, ELSD Education Loan)
- No scheme application submission (guidance only)
- No payment processing

### Assumptions
- Users have smartphones with Telegram (WhatsApp integration planned for post-MVP)
- Users are comfortable with voice messages
- Government scheme data is publicly available
- CSC operators will cooperate with handoff process

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Scheme data becomes outdated | High | Medium | Monthly refresh, crowd-sourced updates |
| LLM hallucination | High | Medium | Ground all responses in database |
| High latency affects UX | Medium | Medium | Parallel API calls, caching |
| Bhashini API downtime | Medium | Low | Fallback to text-only mode |
| Document info becomes outdated | Medium | High | CSC feedback loop, user corrections |

## Dependencies

### External Services
- Telegram Bot API (primary messaging channel)
- Sarvam AI APIs (Saaras v3 STT, Bulbul v3 TTS — primary voice)
- Bhashini (AI4Bharat) APIs (fallback voice)
- AWS Bedrock Nova 2 Lite (primary LLM)
- xAI Grok via OpenAI-compatible API (fallback LLM)
- Jina AI (primary embeddings, jina-embeddings-v3, 1024-dim)
- Voyage AI (fallback embeddings, voyage-multilingual-2)
- AWS services (Lambda, DynamoDB, S3, API Gateway, SQS, CloudWatch)

### Data Sources
- Delhi government scheme database
- Government office directory
- CSC directory
- Document requirement specifications

## Responsible AI

### Bias Mitigation
- Eligibility decisions are driven by structured rules (age, income, category) from official scheme criteria, not LLM inference. This prevents the LLM from introducing demographic bias into eligibility outcomes.
- Scheme data is sourced from official government portals and verified monthly. The system does not generate or infer scheme information.
- Life event classification is validated against a labeled dataset of 100+ examples per category to detect systematic misclassification across demographic groups.

### Explainability
- Every scheme recommendation includes the specific eligibility criteria the user matched (e.g., "You qualify because: age 28, income under ₹2.5L, SC category").
- Rejection prevention warnings cite the specific rule and source (e.g., "SDM certificate required per Delhi Govt notification dated...").
- The system never presents LLM-generated eligibility conclusions as authoritative. All eligibility logic flows through the rules engine.

### Anti-Hallucination Design
- LLM is used for natural language understanding (intent classification, entity extraction, conversational response generation) — not for generating scheme facts.
- All scheme details, eligibility criteria, document requirements, and rejection rules are served from a curated, verified database.
- If the system cannot match a user query to a known scheme or document in the database, it explicitly says so and offers CSC handoff rather than generating a plausible-sounding answer.

### Limitations (Explicit)
- The system does not submit applications on the user's behalf (legal and security constraints).
- The system cannot guarantee scheme approval — it reduces rejection risk, not eliminates it.
- The system does not replace CSCs — it helps users arrive prepared.
- Schemes requiring physical verification (housing inspection, etc.) are flagged but cannot be fully guided through until verification is complete.
- Voice transcription accuracy degrades in high-noise environments. The system detects low-confidence transcriptions and falls back to text mode.

## Offline and Low-Bandwidth Considerations

### Low-Bandwidth Design
- Telegram chosen as primary channel because it works on slow connections, compresses media, and has free Bot API with no business verification delays.
- Voice messages are compressed to OGG Opus format (Telegram default) — typically 8-16 kbps, allowing a 30-second message in ~60KB.
- Text responses are prioritized over audio responses when network quality is poor.
- Scheme data is served as structured text, not images or PDFs, to minimize payload size.

### Offline Fallback
- If external APIs (Sarvam AI, Bhashini, LLM) are unreachable, the system degrades to rule-based keyword matching and pre-written Hindi response templates.
- Frequently accessed scheme information is cached in ElastiCache, so core scheme lookup works even during partial outages.
- CSC handoff (connecting user to a human operator) is always available as the ultimate offline fallback.

## Compliance Requirements

### Privacy
- No permanent storage of personal information
- 24-hour TTL on all session data (balances privacy with multi-day user journeys through bureaucratic processes)
- User consent for voice processing

### Accessibility
- Voice-first design for low-literacy users
- Multi-language support (Hindi, English, Hinglish)
- Fallback text mode for hearing-impaired users

### Government Guidelines
- Compliance with Digital India initiatives
- Adherence to data localization requirements
- Integration with government-approved AI services (Bhashini)