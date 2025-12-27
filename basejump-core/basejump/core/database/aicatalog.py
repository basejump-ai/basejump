"""Catalog of all of the AIs Basejump uses"""

from typing import Optional

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import schemas as sch
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.bedrock_converse import BedrockConverse

logger = set_logging(handler_option="stream", name=__name__)


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

    def get_embedding_model(self, model_info: sch.AzureModelInfo) -> BaseEmbedding:
        """
        The embedding model you want to use throughout the app.
        Currently only AzureOpenAI embedding is supported.
        """
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        return AzureOpenAIEmbedding(
            model=model_info.model_name.value,
            deployment_name=model_info.endpoint_info.deployment_name,
            api_key=model_info.endpoint_info.api_key,
            azure_endpoint=model_info.endpoint_info.endpoint,
            api_version=model_info.api_version,
            callback_manager=self.callback_manager,
        )

    def get_settings(
        self, llm: FunctionCallingLLM, embedding_model_info: sch.AzureModelInfo
    ) -> Settings:  # type:ignore
        """Get a llama index settings object

        Parameters
        ----------
        embedding_model_info
            This will need to be updated to ModelInfo as soon as support for other embeddings
            is included.
        """
        Settings.callback_manager = llm.callback_manager
        Settings.llm = llm
        Settings.embed_model = self.get_embedding_model(model_info=embedding_model_info)
        return Settings

    def get_aws_llm(self, model_info: sch.AWSModelInfo) -> FunctionCallingLLM:
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        return BedrockConverse(
            model=model_info.model_name.value,
            max_tokens=model_info.max_tokens,
            aws_access_key_id=model_info.endpoint_info.access_key,
            aws_secret_access_key=model_info.endpoint_info.secret_access_key,
            region_name=model_info.endpoint_info.deployment_region,
            callback_manager=self.callback_manager,
        )

    def get_azure_llm(self, model_info: sch.AzureModelInfo) -> FunctionCallingLLM:
        assert model_info.endpoint_info, "Missing endpoint info - the pydantic schema should be validating this"
        return AzureOpenAI(
            model=model_info.model_name.value,  # AIModelSchema.GPT4o.value
            temperature=0,
            max_tokens=model_info.max_tokens,
            deployment_name=model_info.endpoint_info.deployment_name,
            api_key=model_info.endpoint_info.api_key,
            azure_endpoint=model_info.endpoint_info.endpoint,
            api_version=model_info.api_version,
            callback_manager=self.callback_manager,
        )
