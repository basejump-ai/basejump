"""Utilities that support the AI functionality or other core business logic within the application"""

import re
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat
from basejump.core.models import constants, enums
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai.formatter import JSONResponseFormatter

logger = set_logging(handler_option="stream", name=__name__)


async def create_prompt_base(
    db: AsyncSession,
    client_user: sch.ClientUserInfo,
    prompt: str,
    model_name: enums.AIModelSchema,
    return_visual_json: bool = True,
) -> sch.PromptMetadataBase:
    """Create prompt metadata before starting to interact with the Agent"""
    prompt_id, prompt_uuid = await crud_chat.create_prompt_history(
        db=db, client_id=client_user.client_id, llm_type=enums.LLMType.DATA_AGENT
    )
    prompt_metadata_base = sch.PromptMetadataBase(
        initial_prompt=prompt,
        user_uuid=client_user.user_uuid,
        user_id=client_user.user_id,
        client_uuid=client_user.client_uuid,
        client_id=client_user.client_id,
        prompt_uuid=prompt_uuid,
        prompt_id=prompt_id,
        llm_type=enums.LLMType.DATA_AGENT,
        model_name=model_name,
        prompt_time=datetime.now(),
        return_visual_json=return_visual_json,
        user_role=client_user.user_role,
    )
    return prompt_metadata_base


async def get_title_description(
    db: AsyncSession,
    prompt_metadata: sch.PromptMetadata,
    sql_query: str,
    query_result: str,
    small_model_info: sch.ModelInfo,
) -> fmt.DescriptionFormat:
    prompt = f"""\
Summarize the following query results into a title and description. \
You will be given the original user prompt, the SQL query to answer the prompt, \
and the query results. DO NOT use any numbers or specific values in the title or description. \n
Prompt: {prompt_metadata.initial_prompt}\n
SQL Query: {sql_query}\n
SQL Results: {query_result}\n
    """
    format_json_response = JSONResponseFormatter(
        response=prompt, pydantic_format=fmt.DescriptionFormat, small_model_info=small_model_info
    )
    return await format_json_response.format()


async def contextualize_prompt(
    recent_interactions: list[sch.MessagePair],
    small_model_info: sch.ModelInfo,
) -> str:
    """Condense multiple prompts into a single contextualized prompt

    Parameters
    ----------
    recent_interactions
        If provided, this will summarize the most recent interactions into a
        single prompt for semantic caching retrieval.
    """
    interactions = ""
    for message in recent_interactions:
        interactions += f"User: {message.prompt.prompt}\n"
        if not message.response:
            response = ""
        else:
            response = message.response.response
        interactions += f"AI: {response}\n"
    format_json_response = JSONResponseFormatter(
        small_model_info=small_model_info,
        response=interactions,
        pydantic_format=fmt.ContextualizedPromptFormat,
    )
    extract = await format_json_response.format()
    prompt = extract.full_context_prompt
    return prompt


def parse_message(text: str) -> list:
    """Clean a message to remove thoughts that don't need to be included in the output"""
    # If already logged, then create a new message UUID since we are logging per SQL query execution
    # logger.info("Webhook messages: %s", text)
    # If webhook is set, then post the thoughts to the webhook
    thoughts = []
    chunks = re.split(r"\n+", text)
    for chunk in chunks:
        # TODO: Make this more robust
        # TODO: Fix the hard reference to structured_sql_generation_tool
        if not chunk:
            continue
        if (
            constants.SQL_TABLES_TOOL_NM_PREFIX in chunk
            or constants.SQL_EXEC_TOOL_NM_PREFIX in chunk
            or constants.VIS_TOOL_NM in chunk
            or constants.DOCS_TOOL_NM in chunk
        ):
            continue
        if "The current language" in chunk:
            continue
        if "I need to use a tool" in chunk:
            continue
        if "Option 1:" in chunk:
            continue
        if "UUID" in chunk or "uuid" in chunk:
            continue
        if ">>" in chunk:
            continue
        if "Use the '" in chunk:
            continue
        if "prefix for the plan" in chunk:
            continue
        if "Plan:" in chunk:
            continue
        # if SQL_OPTION_1 in sentence or SQL_OPTION_2_SUFFIX in sentence or SQL_OPTION_3_SUFFIX:
        #     continue
        else:
            thoughts.append(chunk.strip())
    return thoughts
