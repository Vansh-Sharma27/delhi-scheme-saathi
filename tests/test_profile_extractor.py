"""Tests for rule-based profile extraction helpers."""

from src.services.profile_extractor import (
    extract_by_patterns,
    validate_field_response,
    get_validation_re_prompt,
)


class TestCategoryExtraction:
    """Tests for category extraction edge cases."""

    def test_extracts_bare_age_number(self):
        """Bare age replies should still be recoverable."""
        extracted = extract_by_patterns("19")
        assert extracted.get("age") == 19

    def test_extracts_bare_income_number(self):
        """Bare income replies should still be recoverable."""
        extracted = extract_by_patterns("400000")
        assert extracted.get("annual_income") == 400000

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


class TestFieldValidation:
    """Tests for input validation when asking for profile fields."""

    def test_valid_age_bare_number(self):
        """Valid age as bare number should pass validation."""
        is_valid, error = validate_field_response("age", "25", {})
        assert is_valid is True
        assert error is None

    def test_invalid_age_out_of_range(self):
        """Age > 120 should fail validation."""
        is_valid, error = validate_field_response("age", "200", {})
        assert is_valid is False
        assert error == "invalid_age"

    def test_invalid_age_zero(self):
        """Age = 0 should fail validation."""
        is_valid, error = validate_field_response("age", "0", {})
        assert is_valid is False
        assert error == "invalid_age"

    def test_already_extracted_age_passes(self):
        """If age was already extracted, validation should pass."""
        is_valid, error = validate_field_response("age", "hello", {"age": 25})
        assert is_valid is True

    def test_unrelated_message_passes_validation(self):
        """Unrelated messages should not trigger validation errors."""
        is_valid, error = validate_field_response("age", "I need housing help", {})
        assert is_valid is True

    def test_income_zero_invalid(self):
        """Income = 0 should fail validation."""
        is_valid, error = validate_field_response("annual_income", "0", {})
        assert is_valid is False
        assert error == "invalid_income"


class TestValidationRePrompts:
    """Tests for validation re-prompt generation."""

    def test_invalid_age_reprompt_hindi(self):
        """Invalid age should generate helpful Hindi re-prompt."""
        prompt = get_validation_re_prompt("age", "invalid_age", "hi")
        assert "उम्र" in prompt or "age" in prompt.lower()

    def test_invalid_age_reprompt_english(self):
        """Invalid age should generate helpful English re-prompt."""
        prompt = get_validation_re_prompt("age", "invalid_age", "en")
        assert "age" in prompt.lower()

    def test_invalid_category_lists_options(self):
        """Invalid category re-prompt should list valid options."""
        prompt = get_validation_re_prompt("category", "invalid_category", "en")
        assert "SC" in prompt
        assert "ST" in prompt
        assert "OBC" in prompt
        assert "General" in prompt
        assert "EWS" in prompt


class TestSkipDetection:
    """Tests for 'I don't know' / skip intent detection."""

    def test_skip_detection_import(self):
        """Verify _wants_to_skip can be imported from conversation module."""
        from src.services.conversation import _wants_to_skip
        assert callable(_wants_to_skip)

    def test_skip_dont_know_english(self):
        """'I don't know' should trigger skip."""
        from src.services.conversation import _wants_to_skip
        assert _wants_to_skip("I don't know") is True
        assert _wants_to_skip("i dont know") is True
        assert _wants_to_skip("no idea") is True
        assert _wants_to_skip("not sure") is True

    def test_skip_hindi(self):
        """Hindi skip phrases should trigger skip."""
        from src.services.conversation import _wants_to_skip
        assert _wants_to_skip("pata nahi") is True
        assert _wants_to_skip("nahi pata") is True
        assert _wants_to_skip("पता नहीं") is True

    def test_skip_explicit(self):
        """Explicit skip commands should trigger skip."""
        from src.services.conversation import _wants_to_skip
        assert _wants_to_skip("skip") is True
        assert _wants_to_skip("next") is True
        assert _wants_to_skip("move on") is True

    def test_normal_message_no_skip(self):
        """Normal messages should not trigger skip."""
        from src.services.conversation import _wants_to_skip
        assert _wants_to_skip("I am 25 years old") is False
        assert _wants_to_skip("OBC") is False
        assert _wants_to_skip("housing help") is False
