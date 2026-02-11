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

A voice-first WhatsApp chatbot that goes beyond awareness to provide complete execution assistance for Delhi government schemes. Built on a 100% sovereign AI stack using Bhashini (AI4Bharat) for language processing and AWS Mumbai region for all infrastructure.

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
- AC1.1: System accepts voice messages via WhatsApp
- AC1.2: Voice transcription accuracy > 85% for conversational Hindi
- AC1.3: End-to-end voice response latency < 5 seconds
- AC1.4: Supports Hindi, English, and Hinglish naturally
- AC1.5: Fallback to text mode if voice processing fails
- AC1.6: Uses Bhashini (AI4Bharat) pipeline: IndicWhisper → IndicTrans2 → IndicTTS

**Priority:** High  
**Dependencies:** Bhashini API integration

### FR2: Life-Event Based Discovery

**User Story:** As a user facing a life situation, I want to describe my situation naturally so that the system can identify relevant schemes without me knowing scheme names.

**Acceptance Criteria:**
- AC2.1: Correctly identify life event from natural language with >90% accuracy
- AC2.2: Map each life event to 5-15 relevant schemes
- AC2.3: Handle ambiguous queries with clarifying questions
- AC2.4: Support life event categories:
  - JOB_LOSS, PREGNANCY, CHILDBIRTH, MARRIAGE
  - DISABILITY, ACCIDENT, DEATH_IN_FAMILY  
  - EDUCATION, BUSINESS_STARTUP, RETIREMENT
  - HOUSING, AGRICULTURE (limited for Delhi)

**Example Mappings:**
- "Mera accident ho gaya" → Disability schemes
- "Ghar mein baccha aaya" → Maternity schemes
- "Beti ki shaadi hai" → Marriage assistance
- "Papa retire ho gaye" → Pension schemes

**Priority:** High  
**Dependencies:** NLP model training, scheme database

### FR3: Conversational Eligibility Profiling

**User Story:** As a user, I want to provide my details through natural conversation so that I don't have to fill complex forms.

**Acceptance Criteria:**
- AC3.1: Collect minimum required attributes in 3-4 conversational turns
- AC3.2: Don't re-ask attributes already provided in session
- AC3.3: Handle corrections gracefully ("Actually, my age is 35, not 30")
- AC3.4: Required attributes: age, gender, annual_income, caste_category, domicile_status
- AC3.5: Conditional attributes: employment_status, education_level, marital_status
- AC3.6: Optional attributes: disability_status, bpl_status, minority_status

**Priority:** High  
**Dependencies:** Conversation state management

### FR4: Hybrid Scheme Retrieval

**User Story:** As a user, I want to get the most relevant schemes for my situation so that I don't waste time on irrelevant options.

**Acceptance Criteria:**
- AC4.1: Two-stage matching: structured filtering + semantic search
- AC4.2: Stage 1 SQL filtering by age, income, category, life event
- AC4.3: Stage 2 semantic ranking using embeddings and cosine similarity
- AC4.4: Return top 5 schemes ranked by relevance
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
- AC5.3: Delhi office database with 100+ government offices
- AC5.4: Distance calculation from user's approximate location
- AC5.5: Online portal links verified and working
- AC5.6: Document info available for 30+ common government documents covering all 25 Phase 1 schemes

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
- AC7.1: Walkthrough available for 25 schemes (Phase 1), expanding to 100
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
- AC9.1: Session persists for 24 hours from last message (target users may take days to gather documents between interactions)
- AC9.2: Context maintained across 20+ conversational turns via message sliding window plus periodic LLM-generated conversation summary
- AC9.3: Graceful session expiry handling with context restoration
- AC9.4: No permanent data storage for privacy (24-hour TTL auto-deletes all session data)
- AC9.5: Session includes:
  - User profile (collected attributes)
  - Message history (last 10 messages as sliding window)
  - Conversation summary (LLM-generated every 5 turns to preserve early context)
  - Conversation state
  - Discussed schemes (avoid repetition)
  - Language preference

**Conversation States:**
1. GREETING - Initial welcome
2. SITUATION_UNDERSTANDING - User describes life event
3. PROFILE_COLLECTION - Gathering eligibility attributes
4. SCHEME_MATCHING - Running retrieval
5. SCHEME_PRESENTATION - Showing eligible schemes
6. SCHEME_DETAILS - Deep dive into one scheme
7. DOCUMENT_GUIDANCE - Explaining document requirements
8. REJECTION_WARNINGS - Highlighting common mistakes
9. APPLICATION_HELP - Step-by-step guidance
10. CSC_HANDOFF - Connecting to human help

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
- No permanent conversation storage (24-hour TTL on all session data)
- Rate limiting: 50 messages/hour/user
- Input sanitization for prompt injection prevention

### NFR3: Availability
- 99.5% uptime target
- Graceful degradation if Bhashini unavailable (text-only mode)
- Fallback mechanisms for all external dependencies

### NFR4: Scalability
- Serverless auto-scaling architecture
- Database read replicas for high load
- CDN for static assets and audio files

### NFR5: Data Sovereignty (100% Sovereign AI Stack)
- All data stored in AWS Mumbai region (ap-south-1)
- Voice processing through Bhashini (AI4Bharat, IIT Madras) — government-operated, India-hosted
- No user data export outside India
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
- Must use WhatsApp Business API (primary channel)
- Must integrate with Bhashini for Hindi voice processing
- Must comply with Indian data localization requirements

### Business Constraints
- Focus on Delhi schemes only (MVP scope)
- No scheme application submission (guidance only)
- No payment processing

### Assumptions
- Users have smartphones with WhatsApp
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
- WhatsApp Business API
- Bhashini (AI4Bharat) APIs
- LLM API (Claude/GPT-4/Gemini)
- AWS services (Lambda, DynamoDB, S3, API Gateway)

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
- WhatsApp chosen as primary channel specifically because it works on 2G connections and compresses media automatically.
- Voice messages are compressed to OGG Opus format (WhatsApp default) — typically 8-16 kbps, allowing a 30-second message in ~60KB.
- Text responses are prioritized over audio responses when network quality is poor.
- Scheme data is served as structured text, not images or PDFs, to minimize payload size.

### Offline Fallback
- If external APIs (Bhashini, LLM) are unreachable, the system degrades to rule-based keyword matching and pre-written Hindi response templates.
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