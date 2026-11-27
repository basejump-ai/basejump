"""Token pricing for models served by Amazon Bedrock. These tests need no database."""

from decimal import Decimal

import pytest

from basejump.core.models import enums
from basejump.core.models.ai import token_price

GPT_OSS = enums.AIModelSchema.GPT_OSS_120B.value
COHERE_V4 = enums.AIModelSchema.COHERE_EMBED_V4.value
LLM = enums.AIModelType.LLM
EMBEDDING = enums.AIModelType.EMBEDDING


@pytest.fixture
def warnings(monkeypatch):
    """Clear the configured prices and the warn-once memory, and collect logged warnings."""
    logged: list[str] = []
    monkeypatch.setattr(token_price.logger, "warning", logged.append)
    monkeypatch.setattr(token_price, "_warned_missing_prices", set())
    for name in (
        "bedrock_llm_input_price_per_1k",
        "bedrock_llm_output_price_per_1k",
        "bedrock_embedding_price_per_1k",
    ):
        monkeypatch.setattr(token_price.settings, name, None)
    return logged


def test_configured_llm_prices_are_used_without_warning(monkeypatch, warnings):
    monkeypatch.setattr(token_price.settings, "bedrock_llm_input_price_per_1k", Decimal("0.002"))
    monkeypatch.setattr(token_price.settings, "bedrock_llm_output_price_per_1k", Decimal("0.008"))

    assert token_price.get_model_cost(GPT_OSS, LLM) == (Decimal("0.002"), Decimal("0.008"), GPT_OSS, "AWS Bedrock")
    assert warnings == []


def test_configured_prices_apply_to_any_bedrock_model(monkeypatch, warnings):
    """A Bedrock ID that contains an Azure model name is still priced and recorded as Bedrock."""
    monkeypatch.setattr(token_price.settings, "bedrock_llm_input_price_per_1k", Decimal("0.002"))
    monkeypatch.setattr(token_price.settings, "bedrock_llm_output_price_per_1k", Decimal("0.008"))
    model = "global.openai.gpt-6-astra"

    assert token_price.get_model_cost(model, LLM) == (Decimal("0.002"), Decimal("0.008"), model, "AWS Bedrock")
    assert warnings == []


def test_unset_llm_prices_default_with_one_warning_each(warnings):
    expected = (
        enums.DefaultTokenPrices.GPT_OSS_120B_input.value,
        enums.DefaultTokenPrices.GPT_OSS_120B_output.value,
        GPT_OSS,
        "AWS Bedrock",
    )

    assert token_price.get_model_cost(GPT_OSS, LLM) == expected
    assert token_price.get_model_cost(GPT_OSS, LLM) == expected  # second call must not warn again

    assert len(warnings) == 2
    assert "BASEJUMP_BEDROCK_LLM_INPUT_PRICE_PER_1K is not set" in warnings[0]
    assert "BASEJUMP_BEDROCK_LLM_OUTPUT_PRICE_PER_1K is not set" in warnings[1]


def test_one_unset_llm_price_defaults_only_that_one(monkeypatch, warnings):
    monkeypatch.setattr(token_price.settings, "bedrock_llm_input_price_per_1k", Decimal("0.002"))

    input_price, output_price, _, _ = token_price.get_model_cost(GPT_OSS, LLM)

    assert input_price == Decimal("0.002")
    assert output_price == enums.DefaultTokenPrices.GPT_OSS_120B_output.value
    assert len(warnings) == 1 and "OUTPUT" in warnings[0]


def test_embedding_price_configured_and_default(monkeypatch, warnings):
    assert token_price.get_model_cost(COHERE_V4, EMBEDDING) == (
        enums.DefaultTokenPrices.COHERE_EMBED_V4.value,
        Decimal(0),
        COHERE_V4,
        "AWS Bedrock",
    )
    assert len(warnings) == 1 and "BASEJUMP_BEDROCK_EMBEDDING_PRICE_PER_1K is not set" in warnings[0]

    monkeypatch.setattr(token_price.settings, "bedrock_embedding_price_per_1k", Decimal("0.0001"))
    assert token_price.get_model_cost(COHERE_V4, EMBEDDING)[0] == Decimal("0.0001")
    assert len(warnings) == 1


def test_bedrock_model_without_price_or_default_falls_back_to_gpt41(warnings):
    """Unchanged outcome for Bedrock models that were never priced (Claude Sonnet): GPT-4.1 pricing, with a warning."""
    cost = token_price.get_model_cost("us.anthropic.claude-3-7-sonnet-20250219-v1:0", LLM)

    assert cost == (
        enums.DefaultTokenPrices.GPT41_input.value,
        enums.DefaultTokenPrices.GPT41_output.value,
        enums.AIModelSchema.GPT41.value,
        "Azure OpenAI",
    )
    assert len(warnings) == 1 and "No token price is configured" in warnings[0]


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-6-astra", "gpt-5.2-codex", "text-embedding-3-small"])
def test_azure_model_names_are_not_bedrock(model):
    assert not token_price.is_bedrock_model(model)


def test_price_settings_are_read_from_the_environment(monkeypatch):
    from basejump.core.common.config.settings import Settings

    monkeypatch.setenv("BASEJUMP_BEDROCK_LLM_INPUT_PRICE_PER_1K", "0.00015")
    monkeypatch.setenv("BASEJUMP_BEDROCK_EMBEDDING_PRICE_PER_1K", "0.00012")

    loaded = Settings()

    assert loaded.bedrock_llm_input_price_per_1k == Decimal("0.00015")
    assert loaded.bedrock_llm_output_price_per_1k is None
    assert loaded.bedrock_embedding_price_per_1k == Decimal("0.00012")
