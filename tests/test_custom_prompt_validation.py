"""
Test custom prompt validation and model parameter extraction.
"""

from vtk_prompt.client import VTKPromptClient


class TestCustomPromptValidation:
    """Test validation of custom prompt model parameters."""

    def test_valid_model_and_params(self):
        """Test that valid model and parameters are accepted."""
        client = VTKPromptClient(verbose=False)

        # Use gpt-4.1 which supports temperature
        custom_prompt = {
            "model": "openai/gpt-4.1",
            "modelParameters": {
                "temperature": 0.7,
                "max_tokens": 5000,
            },
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert model == "openai/gpt-4.1"
        assert temp == 0.7
        assert max_tokens == 5000
        assert len(warnings) == 0

    def test_invalid_model_format(self):
        """Test that invalid model format generates warning and uses default."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "model": "gpt-5",  # Missing provider prefix
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert model == "openai/gpt-4o"  # Falls back to default
        assert len(warnings) == 1
        assert "Invalid model format" in warnings[0]

    def test_unsupported_provider(self):
        """Test that unsupported provider generates warning."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "model": "unsupported/model-name",
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert model == "openai/gpt-4o"  # Falls back to default
        assert len(warnings) == 1
        assert "Unsupported provider" in warnings[0]

    def test_model_not_in_curated_list(self):
        """Test that model not in curated list generates warning and uses provider default."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "model": "openai/gpt-99-ultra",  # Not in curated list
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert model == "openai/gpt-5"  # Falls back to provider default
        assert len(warnings) == 1
        assert "not in curated list" in warnings[0]

    def test_temperature_out_of_range(self):
        """Test that temperature out of range generates warning."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "modelParameters": {
                "temperature": 3.0,  # Out of range [0.0, 2.0]
            },
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert temp == 0.5  # Falls back to default
        assert len(warnings) == 1
        assert "out of range" in warnings[0]

    def test_invalid_temperature_type(self):
        """Test that invalid temperature type generates warning."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "modelParameters": {
                "temperature": "hot",  # Invalid type
            },
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert temp == 0.5  # Falls back to default
        assert len(warnings) == 1
        assert "Invalid temperature value" in warnings[0]

    def test_max_tokens_out_of_range(self):
        """Test that max_tokens out of range generates warning."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "modelParameters": {
                "max_tokens": 200000,  # Out of range [1, 100000]
            },
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert max_tokens == 1000  # Falls back to default
        assert len(warnings) == 1
        assert "out of range" in warnings[0]

    def test_temperature_unsupported_by_model(self):
        """Test that temperature warning is generated for models that don't support it."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "model": "openai/gpt-5",  # Doesn't support temperature
            "modelParameters": {
                "temperature": 0.7,
            },
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        assert model == "openai/gpt-5"
        assert temp == 1.0  # Forced to 1.0
        assert len(warnings) == 1
        assert "does not support temperature control" in warnings[0]

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors all generate warnings."""
        client = VTKPromptClient(verbose=False)

        custom_prompt = {
            "model": "invalid-format",
            "modelParameters": {
                "temperature": 5.0,
                "max_tokens": -100,
            },
            "messages": [],
        }

        model, temp, max_tokens, warnings = client._validate_and_extract_model_params(
            custom_prompt, "openai/gpt-4o", 0.5, 1000
        )

        # All should fall back to defaults
        assert model == "openai/gpt-4o"
        assert temp == 0.5
        assert max_tokens == 1000

        # Should have 3 warnings
        assert len(warnings) == 3
        assert any("Invalid model format" in w for w in warnings)
        assert any("out of range" in w for w in warnings)
