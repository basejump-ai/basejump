"""Catalog of all of the AIs Basejump uses"""

from typing import Any, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.llms.openai import OpenAIResponses

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

# Must match the hard-coded "dims" of the Redis vector index schemas
EMBEDDING_DIMENSIONS = 1536


class ReasoningOpenAIResponses(OpenAIResponses):
    """OpenAIResponses for reasoning models that reject sampling parameters.

    llama-index always sends temperature and top_p, but models like gpt-6-astra return
    400 "Unsupported parameter" for them.
    """

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        model_kwargs.pop("temperature", None)
        model_kwargs.pop("top_p", None)
        return model_kwargs


class AICatalog:
    """Organizes all of the LLMs being used into one location"""

    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        self.callback_manager = callback_manager

    def get_llm(self, model_info: sch.ModelInfo) -> FunctionCallingLLM:
        """An LLM used for tasks requiring higher accuracy such as decomposing a question"""

        if isinstance(model_info, sch.AzureModelInfo):
            return self.get_azure_llm(model_info=model_info)
        elif isinstance(model_info, sch.AWSModelInfo):
            return self.get_aws_llm(model_info=model_info)
        else:
            raise NotImplementedError("The provided LLM Info type has not been implemented.")

    def get_embedding_model(self, model_info: sch.EmbeddingModelInfo) -> BaseEmbedding:
        """
        The embedding model you want to use throughout the app.
        Azure OpenAI and AWS Bedrock (Cohere Embed v4) embeddings are supported.
        """
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        if isinstance(model_info, sch.AWSModelInfo):
            return self.get_aws_embedding_model(model_info=model_info)
        elif isinstance(model_info, sch.AzureModelInfo):
            return AzureOpenAIEmbedding(
                model=model_info.model_name.value,
                deployment_name=model_info.endpoint_info.deployment_name,
                api_key=model_info.endpoint_info.api_key,
                azure_endpoint=model_info.endpoint_info.endpoint,
                api_version=model_info.api_version,
                callback_manager=self.callback_manager,
            )
        else:
            raise NotImplementedError("The provided embedding model info type has not been implemented.")

    def get_aws_embedding_model(self, model_info: sch.AWSModelInfo) -> BaseEmbedding:
        """A Bedrock embedding model. Without static keys, the default boto3 credential chain is used."""
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        # Imported here so deployments that only use Azure don't need the Bedrock embeddings package installed
        from llama_index.embeddings.bedrock import BedrockEmbedding

        return BedrockEmbedding(
            model_name=model_info.model_id,
            region_name=model_info.endpoint_info.deployment_region,
            aws_access_key_id=model_info.endpoint_info.access_key or None,
            aws_secret_access_key=model_info.endpoint_info.secret_access_key or None,
            additional_kwargs={"output_dimension": EMBEDDING_DIMENSIONS},
            callback_manager=self.callback_manager,
        )

    def get_aws_llm(self, model_info: sch.AWSModelInfo) -> FunctionCallingLLM:
        """A Bedrock LLM. Without static keys, the default boto3 credential chain is used."""
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        return BedrockConverse(
            model=model_info.model_id,
            max_tokens=model_info.max_tokens,
            aws_access_key_id=model_info.endpoint_info.access_key or None,
            aws_secret_access_key=model_info.endpoint_info.secret_access_key or None,
            region_name=model_info.endpoint_info.deployment_region,
            callback_manager=self.callback_manager,
        )

    def get_azure_llm(self, model_info: sch.AzureModelInfo) -> FunctionCallingLLM:
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        if model_info.model_name == enums.AIModelSchema.GPT6_ASTRA:
            # gpt-6-astra only supports function tools through the Responses API, not /chat/completions.
            # Azure's v1 endpoint is OpenAI-compatible, so the deployment name is passed as the model.
            return ReasoningOpenAIResponses(
                model=model_info.endpoint_info.deployment_name,
                max_output_tokens=model_info.max_tokens,
                api_key=model_info.endpoint_info.api_key,
                api_base=f"{model_info.endpoint_info.endpoint.rstrip('/')}/openai/v1/",
                api_version=model_info.api_version,  # unused by the v1 endpoint
                callback_manager=self.callback_manager,
            )
        return AzureOpenAI(
            model=model_info.model_name.value,
            temperature=0,
            max_tokens=model_info.max_tokens,
            deployment_name=model_info.endpoint_info.deployment_name,
            api_key=model_info.endpoint_info.api_key,
            azure_endpoint=model_info.endpoint_info.endpoint,
            api_version=model_info.api_version,
            callback_manager=self.callback_manager,
        )
