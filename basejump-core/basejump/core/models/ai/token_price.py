import json
import re
from decimal import Decimal
from typing import Optional

import requests
from cachetools import TTLCache, cached
from llama_index.core.callbacks.token_counting import TokenCountingEvent

from basejump.core.common.config.logconfig import set_logging
from basejump.core.common.config.settings import settings
from basejump.core.models import enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)
cache = TTLCache(maxsize=100, ttl=60 * 60 * 24)  # type: ignore

# An Amazon Bedrock model ID or inference profile ID, e.g. "openai.gpt-oss-120b-1:0" or "us.cohere.embed-v4:0".
# Azure model names such as "gpt-4.1" don't start with a provider prefix, so they never match.
BEDROCK_MODEL_ID_PATTERN = re.compile(
    r"^(?:(?:us|eu|apac|global)\.)?(?:ai21|amazon|anthropic|cohere|deepseek|meta|mistral|openai)\."
)
# Missing-price warnings already logged, so each is logged once per process and not once per LLM call
_warned_missing_prices: set[str] = set()


def is_bedrock_model(model: str) -> bool:
    return BEDROCK_MODEL_ID_PATTERN.match(model) is not None


def _warn_once(key: str, message: str) -> None:
    if key not in _warned_missing_prices:
        _warned_missing_prices.add(key)
        logger.warning(message)


def get_bedrock_model_cost(model: str, type_: enums.AIModelType) -> Optional[tuple[Decimal, Decimal, str, str]]:
    """Price a model served by Amazon Bedrock.

    Prices come from settings (BASEJUMP_BEDROCK_LLM_INPUT_PRICE_PER_1K, BASEJUMP_BEDROCK_LLM_OUTPUT_PRICE_PER_1K,
    BASEJUMP_BEDROCK_EMBEDDING_PRICE_PER_1K). When a price isn't set, the model's default price is used and a warning
    is logged. Returns None when a price isn't set and the model has no default, so the caller can fall back.
    """
    ai_model_provider = enums.AIModelProvider.AWS.value
    if type_ == enums.AIModelType.EMBEDDING:
        price = settings.bedrock_embedding_price_per_1k
        if price is None:
            if enums.AIModelSchema.COHERE_EMBED_V4.value not in model:
                return None
            price = enums.DefaultTokenPrices.COHERE_EMBED_V4.value
            _warn_once(
                "embedding",
                f"BASEJUMP_BEDROCK_EMBEDDING_PRICE_PER_1K is not set. Recording {model} usage at the default price "
                f"of ${price:.6f} per 1k tokens.",
            )
        return price, Decimal(0), model, ai_model_provider

    input_price = settings.bedrock_llm_input_price_per_1k
    output_price = settings.bedrock_llm_output_price_per_1k
    if input_price is None or output_price is None:
        if enums.AIModelSchema.GPT_OSS_120B.value not in model:
            return None
        if input_price is None:
            input_price = enums.DefaultTokenPrices.GPT_OSS_120B_input.value
            _warn_once(
                "llm_input",
                f"BASEJUMP_BEDROCK_LLM_INPUT_PRICE_PER_1K is not set. Recording {model} input tokens at the default "
                f"price of ${input_price:.6f} per 1k tokens.",
            )
        if output_price is None:
            output_price = enums.DefaultTokenPrices.GPT_OSS_120B_output.value
            _warn_once(
                "llm_output",
                f"BASEJUMP_BEDROCK_LLM_OUTPUT_PRICE_PER_1K is not set. Recording {model} output tokens at the default "
                f"price of ${output_price:.6f} per 1k tokens.",
            )
    return input_price, output_price, model, ai_model_provider


# Originally taken from here:
# (https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/
# azure-retail-prices?view=rest-aiservices-accountmanagement-2023-05-01#api-endpoint
@cached(cache)
def get_azure_pricing(query: enums.AzurePricingQueries) -> Decimal:
    try:
        # TODO: API version needs to be based on the model being passed in
        API_URL = "https://prices.azure.com/api/retail/prices?api-version=2023-01-01-preview"
        response = requests.get(API_URL, params={"$filter": query.value})
        json_data = json.loads(response.text)
        if len(json_data["Items"]) > 1:
            logger.warning("More than 1 item found for that Azure query. Defaulting to the first item")
        price = json_data["Items"][0]["retailPrice"]
        assert price
    # If there is an error, then default to the default manually set prices
    except Exception:
        logger.warning("Azure price not found. Defaulting to latest GPT-41 price")
        if query == enums.AzurePricingQueries.GPT4o_input:
            price = enums.DefaultTokenPrices.GPT4o_input.value
        elif query == enums.AzurePricingQueries.GPT4o_output:
            price = enums.DefaultTokenPrices.GPT4o_output.value
        elif query == enums.AzurePricingQueries.ADA:
            price = enums.DefaultTokenPrices.ADA.value
        else:
            price = enums.DefaultTokenPrices.GPT41_output.value
    return price


def get_model_cost(model: Optional[str], type_: enums.AIModelType) -> tuple[Decimal, Decimal, str, str]:
    DEFAULT_LLM_MODEL = enums.AIModelSchema.GPT41.value
    DEFAULT_EMBED_MODEL = enums.AIModelSchema.ADA.value
    DEFAULT_MODEL = DEFAULT_EMBED_MODEL if type_ == enums.AIModelType.EMBEDDING else DEFAULT_LLM_MODEL

    if not model:
        logger.warning(f"Missing model needed for token count. Defaulting to {DEFAULT_MODEL}")
        return get_model_cost(model=DEFAULT_MODEL, type_=type_)
    # Bedrock models are checked first: a Bedrock ID can contain an Azure model name ("openai.gpt-6-astra").
    if is_bedrock_model(model):
        bedrock_cost = get_bedrock_model_cost(model=model, type_=type_)
        if bedrock_cost is not None:
            return bedrock_cost
        _warn_once(
            f"no_price:{model}",
            f"No token price is configured for Bedrock model {model}, and it has no default price. Set the "
            f"BASEJUMP_BEDROCK_*_PRICE_PER_1K settings. Defaulting to {DEFAULT_MODEL} pricing.",
        )
        return get_model_cost(model=DEFAULT_MODEL, type_=type_)
    # HACK: Replacing periods in GPT4.1 with nothing to match model from llama index
    if enums.AIModelSchema.GPT6_ASTRA.value in model:
        # TODO: Implement the azure pricing queries instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT6_ASTRA_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT6_ASTRA_output.value
        model = enums.AIModelSchema.GPT6_ASTRA.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT52_CODEX.value in model:
        # TODO: Implement the azure pricing queries instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT52_CODEX_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT52_CODEX_output.value
        model = enums.AIModelSchema.GPT52_CODEX.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT52.value in model:
        # TODO: Implement the azure pricing queries instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT52_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT52_output.value
        model = enums.AIModelSchema.GPT52.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT51_CODEX_MAX.value in model:
        # TODO: Implement the azure pricing queries instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT51_CODEX_MAX_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT51_CODEX_MAX_output.value
        model = enums.AIModelSchema.GPT51_CODEX_MAX.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT51.value in model:
        # TODO: Implement the azure pricing queries instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT51_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT51_output.value
        model = enums.AIModelSchema.GPT51.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT5.value in model:
        # TODO: Implement the azure pricing queries instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT5_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT5_output.value
        model = enums.AIModelSchema.GPT5.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT41.value in model:
        # TODO: Implement the azure pricing queries for GPT 4.1 instead of the manually input default token prices
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GPT41_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GPT41_output.value
        model = enums.AIModelSchema.GPT41.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GPT4o.value in model:
        cost_per_1k_tokens_input = get_azure_pricing(query=enums.AzurePricingQueries.GPT4o_input)
        cost_per_1k_tokens_output = get_azure_pricing(query=enums.AzurePricingQueries.GPT4o_output)
        model = enums.AIModelSchema.GPT4o.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.ADA3_SMALL.value in model:
        # TODO: Add azure pricing query for ADA3 small
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.ADA3_SMALL.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.ADA3_SMALL.value
        model = enums.AIModelSchema.ADA3_SMALL.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.ADA.value in model:
        cost_per_1k_tokens_input = get_azure_pricing(query=enums.AzurePricingQueries.ADA)
        cost_per_1k_tokens_output = Decimal(0)
        model = enums.AIModelSchema.ADA.value
        ai_model_provider = enums.AIModelProvider.AZURE_OPENAI.value
    elif enums.AIModelSchema.GROQ.value in model:
        # TODO: Check if Groq has an API for real-time API costs
        cost_per_1k_tokens_input = enums.DefaultTokenPrices.GROQ_70B_llama_input.value
        cost_per_1k_tokens_output = enums.DefaultTokenPrices.GROQ_70B_llama_output.value
        model = enums.AIModelSchema.GROQ.value
        ai_model_provider = enums.AIModelProvider.GROQ.value

    else:
        logger.warning(f"AI Model {model} not implemented. Defaulting to {DEFAULT_MODEL}")
        return get_model_cost(model=DEFAULT_MODEL, type_=type_)

    return cost_per_1k_tokens_input, cost_per_1k_tokens_output, model, ai_model_provider


def get_token_count_obj(
    token_count: TokenCountingEvent, prompt_metadata: sch.PromptMetadata, type_: enums.AIModelType
) -> sch.TokenCountSchema:
    cost_per_1k_tokens_input, cost_per_1k_tokens_output, model, ai_model_provider = get_model_cost(
        model=token_count.model, type_=type_
    )
    # Get the cost of the tokens
    token_count_obj = sch.TokenCountSchema(
        token_uuid=str(token_count.event_id),
        prompt=prompt_metadata.initial_prompt,
        prompt_id=prompt_metadata.prompt_id,
        client_id=prompt_metadata.client_id,
        ai_model_provider=ai_model_provider,
        ai_model_nm=model,
        cost_per_1k_tokens_input=cost_per_1k_tokens_input,
        cost_per_1k_tokens_output=cost_per_1k_tokens_output,
        total_embedding_token_count=token_count.total_token_count if type_ == enums.AIModelType.EMBEDDING else 0,
        prompt_llm_token_count=token_count.prompt_token_count,
        completion_llm_token_count=token_count.completion_token_count,
        total_llm_token_count=token_count.total_token_count,
    )

    return token_count_obj
