"""Tests for rule-based profile extraction helpers."""

from src.services.profile_extractor import extract_by_patterns


class TestCategoryExtraction:
    """Tests for category extraction edge cases."""

    def test_does_not_extract_st_from_study_word(self):
        """Words like 'study' must not map to ST category."""
        extracted = extract_by_patterns(
            "I need education schemes for my son and want to study in India."
        )
        assert extracted.get("category") is None

    def test_does_not_extract_sc_from_schemes_word(self):
        """Words like 'schemes' must not map to SC category."""
        extracted = extract_by_patterns("Please suggest welfare schemes for me.")
        assert extracted.get("category") is None

    def test_extracts_sc_when_explicit(self):
        """Explicit SC mention should be extracted."""
        extracted = extract_by_patterns("I belong to SC category.")
        assert extracted.get("category") == "SC"

    def test_extracts_st_when_explicit(self):
        """Explicit ST mention should be extracted."""
        extracted = extract_by_patterns("My caste is ST.")
        assert extracted.get("category") == "ST"
