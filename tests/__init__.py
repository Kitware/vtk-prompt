"""
VTK Prompt test suite.

This package contains comprehensive tests for the VTK Prompt system,
including smoke tests, integration tests, and unit tests.

Test Categories:
- test_client_basic.py: Basic VTKPromptClient functionality
- test_provider_smoke.py: Provider/model combination smoke tests
- test_rag_integration.py: RAG system integration tests
- test_provider_utils.py: Provider utility function tests
- conftest.py: Shared fixtures and test configuration

To run tests:
    pytest tests/                    # Run all tests
    pytest tests/test_provider_smoke.py  # Run specific test file
    pytest -v                        # Verbose output
    pytest -x                        # Stop on first failure
"""
