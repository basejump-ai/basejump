"""Utilities that support the AI functionality or other core business logic within the application"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat
from basejump.core.models import enums
from basejump.core.models import schemas as sch

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
