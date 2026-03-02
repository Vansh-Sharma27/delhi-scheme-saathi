"""Test configuration and fixtures."""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_user_profile() -> dict:
    """Sample user profile for testing."""
    return {
        "age": 58,
        "gender": "female",
        "category": "SC",
        "annual_income": 80000,
        "employment_status": "unemployed",
        "marital_status": "widowed",
        "life_event": "DEATH_IN_FAMILY",
        "district": "North Delhi",
    }


@pytest.fixture
def sample_scheme() -> dict:
    """Sample scheme data for testing."""
    return {
        "id": "SCH-TEST-001",
        "name": "Test Widow Pension",
        "name_hindi": "परीक्षण विधवा पेंशन",
        "department": "Social Welfare",
        "department_hindi": "समाज कल्याण विभाग",
        "level": "state",
        "description": "Pension for widows",
        "description_hindi": "विधवाओं के लिए पेंशन",
        "benefits_amount": 250000,
        "eligibility": {
            "min_age": 18,
            "max_age": None,
            "genders": ["female"],
            "categories": ["SC", "ST", "OBC", "General"],
            "max_income": 100000,
        },
        "life_events": ["DEATH_IN_FAMILY"],
        "documents_required": ["DOC-AADHAAR", "DOC-DEATH-CERT"],
        "rejection_rules": ["RULE-TEST-001"],
        "is_active": True,
    }


@pytest.fixture
def sample_document() -> dict:
    """Sample document data for testing."""
    return {
        "id": "DOC-AADHAAR",
        "name": "Aadhaar Card",
        "name_hindi": "आधार कार्ड",
        "issuing_authority": "UIDAI",
        "online_portal": "https://uidai.gov.in",
        "prerequisites": [],
        "fee": "Free",
        "processing_time": "5-7 days",
    }


@pytest.fixture
def sample_office() -> dict:
    """Sample office data for testing."""
    return {
        "id": "OFF-TEST-001",
        "name": "Test CSC Center",
        "type": "CSC",
        "address": "Test Address, Delhi",
        "district": "North Delhi",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "phone": "1234567890",
        "working_hours": "9 AM - 5 PM",
        "services": ["DOC-AADHAAR", "DOC-INCOME-CERT"],
    }
