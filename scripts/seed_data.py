"""Seed the database with scheme data from JSON files."""

import asyncio
import json
import os
from pathlib import Path

import asyncpg


DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/delhi_scheme_saathi"


def resolve_database_url(env_value: str | None) -> str:
    """Return a validated database URL for seeding."""
    from urllib.parse import urlparse

    if env_value is None:
        return DEFAULT_DATABASE_URL

    database_url = env_value.strip().strip('"').strip("'")
    if not database_url:
        raise RuntimeError("DATABASE_URL is empty")
    if not database_url.startswith(("postgresql://", "postgres://")):
        scheme = database_url.split(":", 1)[0] if ":" in database_url else ""
        scheme_display = scheme or "<empty>"
        raise RuntimeError(
            "DATABASE_URL must start with postgres:// or postgresql:// "
            f"(got {scheme_display!r})"
        )
    parsed = urlparse(database_url)
    if not parsed.hostname:
        raise RuntimeError("DATABASE_URL is missing a hostname")
    if "://" in parsed.hostname:
        raise RuntimeError(
            "DATABASE_URL host is malformed. Do not include http:// or https:// inside the hostname."
        )
    return database_url


async def seed_database() -> None:
    """Load all JSON data into PostgreSQL."""
    database_url = resolve_database_url(os.getenv("DATABASE_URL"))

    conn = await asyncpg.connect(database_url)

    try:
        # Load schemes
        schemes_file = DATA_DIR / "all_schemes.json"
        if schemes_file.exists():
            schemes = json.loads(schemes_file.read_text())
            print(f"Loading {len(schemes)} schemes...")

            for scheme in schemes:
                # Convert embedding list to proper format or None
                embedding = scheme.get("description_embedding")
                if embedding and len(embedding) > 0:
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                else:
                    embedding_str = None

                await conn.execute("""
                    INSERT INTO schemes (
                        id, name, name_hindi, department, department_hindi, level,
                        description, description_hindi, benefits_summary, benefits_amount,
                        benefits_frequency, eligibility, description_embedding,
                        documents_required, rejection_rules, application_url,
                        application_steps, offline_process, processing_time, helpline,
                        life_events, tags, official_url, metadata, is_active
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13::vector, $14, $15, $16, $17, $18, $19, $20,
                        $21, $22, $23, $24, $25
                    ) ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        name_hindi = EXCLUDED.name_hindi,
                        department = EXCLUDED.department,
                        department_hindi = EXCLUDED.department_hindi,
                        level = EXCLUDED.level,
                        description = EXCLUDED.description,
                        description_hindi = EXCLUDED.description_hindi,
                        benefits_summary = EXCLUDED.benefits_summary,
                        benefits_amount = EXCLUDED.benefits_amount,
                        benefits_frequency = EXCLUDED.benefits_frequency,
                        eligibility = EXCLUDED.eligibility,
                        description_embedding = EXCLUDED.description_embedding,
                        documents_required = EXCLUDED.documents_required,
                        rejection_rules = EXCLUDED.rejection_rules,
                        application_url = EXCLUDED.application_url,
                        application_steps = EXCLUDED.application_steps,
                        offline_process = EXCLUDED.offline_process,
                        processing_time = EXCLUDED.processing_time,
                        helpline = EXCLUDED.helpline,
                        life_events = EXCLUDED.life_events,
                        tags = EXCLUDED.tags,
                        official_url = EXCLUDED.official_url,
                        metadata = EXCLUDED.metadata,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                """,
                    scheme["id"],
                    scheme["name"],
                    scheme.get("name_hindi", scheme["name"]),
                    scheme["department"],
                    scheme.get("department_hindi", scheme["department"]),
                    scheme.get("level", "central"),
                    scheme["description"],
                    scheme.get("description_hindi", scheme["description"]),
                    scheme.get("benefits_summary"),
                    scheme.get("benefits_amount"),
                    scheme.get("benefits_frequency"),
                    json.dumps(scheme.get("eligibility", {})),
                    embedding_str,
                    scheme.get("documents_required", []),
                    scheme.get("rejection_rules", []),
                    scheme.get("application_url"),
                    scheme.get("application_steps", []),
                    scheme.get("offline_process"),
                    scheme.get("processing_time"),
                    json.dumps(scheme.get("helpline")) if scheme.get("helpline") else None,
                    scheme.get("life_events", []),
                    scheme.get("tags", []),
                    scheme.get("official_url"),
                    json.dumps(scheme.get("metadata", {})),
                    scheme.get("is_active", True),
                )
            print(f"  Loaded {len(schemes)} schemes")

        # Load documents
        docs_file = DATA_DIR / "all_documents.json"
        if docs_file.exists():
            docs = json.loads(docs_file.read_text())
            print(f"Loading {len(docs)} documents...")

            for doc in docs:
                await conn.execute("""
                    INSERT INTO documents (
                        id, name, name_hindi, issuing_authority, alternate_authority,
                        online_portal, prerequisites, fee, fee_bpl, processing_time,
                        validity_period, format_requirements, common_mistakes, delhi_offices
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    ) ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        name_hindi = EXCLUDED.name_hindi,
                        updated_at = NOW()
                """,
                    doc["id"],
                    doc["name"],
                    doc.get("name_hindi", doc["name"]),
                    doc["issuing_authority"],
                    doc.get("alternate_authority"),
                    doc.get("online_portal"),
                    doc.get("prerequisites", []),
                    doc.get("fee"),
                    doc.get("fee_bpl"),
                    doc.get("processing_time"),
                    doc.get("validity_period"),
                    doc.get("format_requirements", []),
                    doc.get("common_mistakes", []),
                    doc.get("delhi_offices", []),
                )
            print(f"  Loaded {len(docs)} documents")

        # Load offices
        offices_file = DATA_DIR / "all_offices.json"
        if offices_file.exists():
            offices = json.loads(offices_file.read_text())
            print(f"Loading {len(offices)} offices...")

            for office in offices:
                await conn.execute("""
                    INSERT INTO offices (
                        id, name, type, address, district, pincode,
                        latitude, longitude, phone, working_hours,
                        services, fee_structure, operator_name
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                    ) ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        address = EXCLUDED.address,
                        updated_at = NOW()
                """,
                    office["id"],
                    office["name"],
                    office["type"],
                    office["address"],
                    office["district"],
                    office.get("pincode"),
                    office.get("latitude"),
                    office.get("longitude"),
                    office.get("phone"),
                    office.get("working_hours"),
                    office.get("services", []),
                    json.dumps(office.get("fee_structure", {})),
                    office.get("operator_name"),
                )
            print(f"  Loaded {len(offices)} offices")

        # Load rejection rules
        rules_file = DATA_DIR / "all_rejection_rules.json"
        if rules_file.exists():
            rules = json.loads(rules_file.read_text())
            print(f"Loading {len(rules)} rejection rules...")

            for rule in rules:
                await conn.execute("""
                    INSERT INTO rejection_rules (
                        id, scheme_id, rule_type, description, description_hindi,
                        severity, prevention_tip, examples
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8
                    ) ON CONFLICT (id) DO UPDATE SET
                        description = EXCLUDED.description,
                        prevention_tip = EXCLUDED.prevention_tip,
                        updated_at = NOW()
                """,
                    rule["id"],
                    rule["scheme_id"],
                    rule["rule_type"],
                    rule["description"],
                    rule.get("description_hindi", rule["description"]),
                    rule["severity"],
                    rule["prevention_tip"],
                    rule.get("examples", []),
                )
            print(f"  Loaded {len(rules)} rejection rules")

        # Verify
        scheme_count = await conn.fetchval("SELECT COUNT(*) FROM schemes")
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        office_count = await conn.fetchval("SELECT COUNT(*) FROM offices")
        rule_count = await conn.fetchval("SELECT COUNT(*) FROM rejection_rules")

        print(f"\nDatabase seeded successfully!")
        print(f"  Schemes: {scheme_count}")
        print(f"  Documents: {doc_count}")
        print(f"  Offices: {office_count}")
        print(f"  Rejection Rules: {rule_count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed_database())
