import json
from decimal import Decimal
from typing import Optional

import requests
from cachetools import TTLCache, cached
from llama_index.core.callbacks.token_counting import TokenCountingEvent

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)
cache = TTLCache(maxsize=100, ttl=60 * 60 * 24)  # type: ignore


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
    # HACK: Replacing periods in GPT4.1 with nothing to match model from llama index
    elif enums.AIModelSchema.GPT41.value in model.replace(".", ""):
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
