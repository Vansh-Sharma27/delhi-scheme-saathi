-- Delhi Scheme Saathi - PostgreSQL Schema
-- Auto-runs on container first start via docker-entrypoint-initdb.d

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Schemes table (core entity)
CREATE TABLE schemes (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    name_hindi VARCHAR(200) NOT NULL,
    department VARCHAR(200) NOT NULL,
    department_hindi VARCHAR(200) NOT NULL,
    level VARCHAR(20) NOT NULL CHECK (level IN ('central', 'state', 'district')),
    description TEXT NOT NULL,
    description_hindi TEXT NOT NULL,
    benefits_summary TEXT,
    benefits_amount INTEGER,
    benefits_frequency VARCHAR(50),
    eligibility JSONB NOT NULL,
    description_embedding vector(1024),
    documents_required TEXT[] DEFAULT '{}',
    rejection_rules TEXT[] DEFAULT '{}',
    application_url VARCHAR(500),
    application_steps TEXT[] DEFAULT '{}',
    offline_process TEXT,
    processing_time VARCHAR(100),
    helpline JSONB,
    life_events TEXT[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    official_url VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    last_verified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Documents table
CREATE TABLE documents (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    name_hindi VARCHAR(200) NOT NULL,
    issuing_authority VARCHAR(200) NOT NULL,
    alternate_authority VARCHAR(200),
    online_portal VARCHAR(500),
    prerequisites TEXT[] DEFAULT '{}',
    fee VARCHAR(100),
    fee_bpl VARCHAR(100),
    processing_time VARCHAR(100),
    validity_period VARCHAR(100),
    format_requirements TEXT[] DEFAULT '{}',
    common_mistakes TEXT[] DEFAULT '{}',
    delhi_offices TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Offices table
CREATE TABLE offices (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50) NOT NULL,
    address TEXT NOT NULL,
    district VARCHAR(50) NOT NULL,
    pincode VARCHAR(10),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    phone VARCHAR(50),
    working_hours VARCHAR(100),
    services TEXT[] DEFAULT '{}',
    fee_structure JSONB DEFAULT '{}',
    operator_name VARCHAR(100),
    last_verified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Rejection rules table
CREATE TABLE rejection_rules (
    id VARCHAR(50) PRIMARY KEY,
    scheme_id VARCHAR(50) REFERENCES schemes(id) ON DELETE CASCADE,
    rule_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    description_hindi TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'warning', 'high')),
    prevention_tip TEXT NOT NULL,
    examples TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Life events taxonomy
CREATE TABLE life_events_taxonomy (
    key VARCHAR(50) PRIMARY KEY,
    display_name VARCHAR(100) NOT NULL,
    display_name_hindi VARCHAR(100) NOT NULL,
    aliases TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_schemes_life_events ON schemes USING GIN (life_events);
CREATE INDEX idx_schemes_eligibility ON schemes USING GIN (eligibility);
CREATE INDEX idx_schemes_tags ON schemes USING GIN (tags);
CREATE INDEX idx_schemes_level ON schemes (level);
CREATE INDEX idx_schemes_is_active ON schemes (is_active);

-- Vector search index (HNSW for cosine similarity)
CREATE INDEX idx_schemes_embedding ON schemes
    USING hnsw (description_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_offices_location ON offices (latitude, longitude);
CREATE INDEX idx_offices_district ON offices (district);
CREATE INDEX idx_offices_type ON offices (type);
CREATE INDEX idx_offices_services ON offices USING GIN (services);

CREATE INDEX idx_rejection_rules_scheme ON rejection_rules (scheme_id);
CREATE INDEX idx_rejection_rules_severity ON rejection_rules (severity);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_schemes_updated_at
    BEFORE UPDATE ON schemes
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_offices_updated_at
    BEFORE UPDATE ON offices
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_rejection_rules_updated_at
    BEFORE UPDATE ON rejection_rules
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- Seed life events taxonomy
INSERT INTO life_events_taxonomy (key, display_name, display_name_hindi, aliases) VALUES
('HOUSING', 'Housing & Property', 'आवास एवं संपत्ति', ARRAY['buying_home', 'constructing_home', 'renting_home', 'property_purchase']),
('MARRIAGE', 'Marriage', 'विवाह', ARRAY['marriage', 'wedding', 'getting_married']),
('CHILDBIRTH', 'Childbirth & Parenting', 'जन्म एवं पालन-पोषण', ARRAY['starting_family', 'child_birth', 'having_children', 'new_baby']),
('EDUCATION', 'Education', 'शिक्षा', ARRAY['education', 'higher_education', 'professional_course', 'studying', 'studying_abroad']),
('HEALTH_CRISIS', 'Health Emergency', 'स्वास्थ्य आपातकाल', ARRAY['illness', 'hospitalization', 'medical_emergency', 'chronic_disease', 'surgery', 'health_issue']),
('DEATH_IN_FAMILY', 'Death in Family', 'परिवार में मृत्यु', ARRAY['death_of_spouse', 'widowhood', 'death_in_family', 'bereavement']),
('MARITAL_DISTRESS', 'Marital Distress', 'वैवाहिक कठिनाई', ARRAY['divorce', 'separation', 'abandonment', 'destitution', 'deserted']),
('JOB_LOSS', 'Job Loss & Unemployment', 'नौकरी छूटना एवं बेरोजगारी', ARRAY['unemployment', 'job_loss', 'retrenchment', 'laid_off']),
('BUSINESS_STARTUP', 'Starting a Business', 'व्यवसाय शुरू करना', ARRAY['starting_business', 'self_employment', 'entrepreneurship', 'skill_utilization', 'business_expansion']),
('WOMEN_EMPOWERMENT', 'Women Empowerment', 'महिला सशक्तिकरण', ARRAY['women_empowerment', 'women_welfare', 'girl_child']);
