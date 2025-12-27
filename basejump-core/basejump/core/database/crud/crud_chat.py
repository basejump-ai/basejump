"""Functions to interact with the database for tables related to the chat.py endpoint module and chat-related tables"""

import copy
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.aicatalog import AICatalog
from basejump.core.database.crud import crud_utils
from basejump.core.database.db_utils import add_message_context
from basejump.core.database.token_price import get_token_count_obj
from basejump.core.database.vector_utils import delete_nodes
from basejump.core.models import constants, enums, models
from basejump.core.models import schemas as sch
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import VectorMemory
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from redis.asyncio import Redis as RedisAsync
from sqlalchemy import Row, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = set_logging(handler_option="stream", name=__name__)


async def get_chat(
    db: AsyncSession,
    chat_uuid: uuid.UUID,
    user_id: int,
    include_all_client_info: bool = False,
    empty_chats_only: Optional[bool] = False,
) -> Optional[models.Chat]:
    if empty_chats_only:
        stmt = select(models.Chat).filter(models.Chat.chat_uuid == chat_uuid, models.Chat.chat_name.is_(None))
    else:
        stmt = select(models.Chat).filter_by(chat_uuid=chat_uuid)
    if not include_all_client_info:
        stmt = stmt.filter(models.Chat.user_id == user_id)
    chat = await db.execute(stmt)
    return chat.scalar_one_or_none()


async def get_chat_from_id(db: AsyncSession, chat_id: int) -> Optional[models.Chat]:
    chat = await db.execute(select(models.Chat).filter_by(chat_id=chat_id))
    return chat.scalar_one_or_none()


async def get_chats_from_client_id(db: AsyncSession, client_id: int) -> list[models.Chat]:
    chats = await db.execute(select(models.Chat).filter_by(client_id=client_id))
    return list(chats.scalars().all())


async def get_prompts_from_client_id(db: AsyncSession, client_id: int) -> list[models.PromptHistory]:
    chats = await db.execute(select(models.PromptHistory).filter_by(client_id=client_id))
    return list(chats.scalars().all())


async def get_chats(db: AsyncSession, user_id: int, empty_chats_only: Optional[bool] = False) -> list[models.Chat]:
    if empty_chats_only:
        result = await db.execute(
            select(models.Chat).filter(models.Chat.user_id == user_id, models.Chat.chat_name.is_(None))
        )
    else:
        result = await db.execute(select(models.Chat).filter(models.Chat.user_id == user_id))

    chats = result.scalars().all()
    return list(chats)


async def get_chat_history_limited(db: AsyncSession, chat_id: int, limit=1) -> Sequence[models.ChatHistory]:
    """Gets chat history for chats that aren't indexed"""
    stmt = (
        select(models.ChatHistory)
        .filter(models.ChatHistory.chat_id == chat_id)
        .order_by(desc(models.ChatHistory.timestamp))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_chat_history_for_ai(db: AsyncSession, chat_id: int) -> list[ChatMessage]:
    """Get the conversation history for a user. This is a list of messages.

    Parameters
    ----------
    chat_id
        The chat ID for the conversation
    """
    stmt = (
        select(
            models.ChatHistory,
            models.VisualResultHistory,
        )
        .join(
            models.VisualResultHistory,
            (
                (models.ChatHistory.visual_result_uuid == models.VisualResultHistory.visual_result_uuid)
                & (models.ChatHistory.role == sch.MessageRole.ASSISTANT.value)
            ),
            isouter=True,
        )
        .filter(models.ChatHistory.chat_id == chat_id)
        .order_by(models.ChatHistory.prompt_time, models.ChatHistory.timestamp)
        .limit(100)
    )
    chat_hist_base = await db.execute(stmt)
    chat_hist = chat_hist_base.all()
    chat_history = []
    for chat, visual_result_hist in chat_hist:
        visual_json = json.loads(visual_result_hist.visual_json) if visual_result_hist else None
        content = add_message_context(
            content=chat.content,
            sql_query=chat.sql_query,
            timestamp=chat.timestamp,
            result_uuid=chat.result_uuid,
            visual_json=visual_json,
        )
        message = ChatMessage(role=chat.role, content=content, timestamp=chat.timestamp)
        chat_history.append(message)
    await db.commit()  # NOTE: Committing to avoid idle in transaction
    return chat_history


async def get_chat_history_from_msgs(db: AsyncSession, msg_uuids: list[uuid.UUID]):
    stmt = select(models.ChatHistory).filter(models.ChatHistory.msg_uuid.in_(msg_uuids))
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_message(db: AsyncSession, msg_uuid: uuid.UUID) -> Optional[models.ChatHistory]:
    message = await db.execute(select(models.ChatHistory).filter_by(msg_uuid=msg_uuid))
    return message.scalar_one_or_none()


async def save_message(
    db: AsyncSession,
    message: sch.APIMessage,
    prompt_metadata: sch.PromptMetadataBase,
    chat_metadata: sch.ChatMetadata,
    query_result: sch.MessageQueryResult,
    msg_in_index: bool = False,
) -> None:
    """Save the conversation history for a client"""
    # Save the chat history
    # Check if the message already exists in the chat history, if so, update it
    stmt = select(models.ChatHistory).filter_by(msg_uuid=message.msg_uuid)
    result_raw = await db.execute(stmt)
    result = result_raw.scalar_one_or_none()
    if result:
        # If there is a match, then just update the message instead
        result.content = message.content
        result.msg_type = message.msg_type.value if message.msg_type else None
        result.internal_content = add_message_context(
            content=message.content,
            timestamp=datetime.now(ZoneInfo("UTC")).isoformat(),
            sql_query=query_result.sql_query,
        )
        result.sql_query = query_result.sql_query
        result.result_uuid = query_result.result_uuid
        result.visual_result_uuid = query_result.visual_result_uuid
        result.result_type = query_result.result_type.value if query_result.result_type else None
    else:
        # Otherwise add a new message in the chat history
        chat_hist = models.ChatHistory(
            client_id=prompt_metadata.client_id,
            msg_uuid=message.msg_uuid,
            msg_in_index=msg_in_index,
            parent_msg_uuid=chat_metadata.parent_msg_uuid,
            chat_id=chat_metadata.chat_id,
            prompt_uuid=prompt_metadata.prompt_uuid,
            initial_prompt=prompt_metadata.initial_prompt,
            prompt_time=prompt_metadata.prompt_time,
            content=message.content,
            internal_content=add_message_context(
                content=message.content,
                timestamp=datetime.now(ZoneInfo("UTC")).isoformat(),
                sql_query=query_result.sql_query,
            ),  # BC v0.26.1
            role=message.role.value if message.role else None,
            msg_type=message.msg_type.value if message.msg_type else None,
            sql_query=query_result.sql_query,
            result_uuid=query_result.result_uuid,
            visual_result_uuid=query_result.visual_result_uuid,
            result_type=query_result.result_type.value if query_result.result_type else None,
            timestamp=datetime.now(ZoneInfo("UTC")),
        )
        db.add(chat_hist)
    await db.commit()
    logger.debug("Completed saving chat message to DB")


async def save_token_counts(
    db: AsyncSession,
    prompt_metadata: sch.PromptMetadata,
):
    # NOTE: Using in since the model version can be appended to the end of the model name
    for token_count in prompt_metadata.token_counter.llm_token_counts:
        token_count_obj = get_token_count_obj(
            token_count=token_count, prompt_metadata=prompt_metadata, type_=enums.AIModelType.LLM
        )
        token_id = await crud_utils.get_next_val(
            db=db, full_table_nm=str(models.TokenCount.__table__), column_nm="token_id"
        )
        token_count_db = models.TokenCount(token_id=token_id, **token_count_obj.dict())
        db.add(token_count_db)
        token_count_assoc = models.TokenUserAssociation(
            client_id=prompt_metadata.client_id, user_id=prompt_metadata.user_id, token_id=token_id
        )
        db.add(token_count_assoc)

    for token_count in prompt_metadata.token_counter.embedding_token_counts:
        token_count_obj = get_token_count_obj(
            token_count=token_count, prompt_metadata=prompt_metadata, type_=enums.AIModelType.EMBEDDING
        )
        token_id = await crud_utils.get_next_val(
            db=db, full_table_nm=str(models.TokenCount.__table__), column_nm="token_id"
        )
        token_count_db = models.TokenCount(token_id=token_id, **token_count_obj.dict())
        db.add(token_count_db)
        token_count_assoc = models.TokenUserAssociation(
            client_id=prompt_metadata.client_id, user_id=prompt_metadata.user_id, token_id=token_id
        )
        db.add(token_count_assoc)
    await db.commit()
    prompt_metadata.token_counter.reset_counts()


async def create_prompt_history(
    db: AsyncSession, client_id: int, llm_type: enums.LLMType, prompt_uuid: Optional[uuid.UUID] = None
) -> tuple[int, uuid.UUID]:
    if prompt_uuid:
        prompt = models.PromptHistory(prompt_uuid=prompt_uuid, client_id=client_id, llm_type=llm_type.value)
    else:
        prompt = models.PromptHistory(client_id=client_id, llm_type=llm_type.value)
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)

    prompt_uuid = copy.copy(prompt.prompt_uuid)
    prompt_id = copy.copy(prompt.prompt_id)
    return prompt_id, prompt_uuid


async def get_prompt_history(db: AsyncSession, prompt_uuid: uuid.UUID) -> Optional[models.PromptHistory]:
    stmt = select(models.PromptHistory).filter_by(prompt_uuid=prompt_uuid)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def delete_chat_msgs_from_vector(
    db: AsyncSession, client_id: int, msg_uuids: list[uuid.UUID], redis_client_async: RedisAsync
) -> None:
    chat_history = await get_chat_history_from_msgs(db=db, msg_uuids=msg_uuids)
    for msg in chat_history:
        msg.msg_in_index = False
    await db.commit()
    await delete_nodes(
        client_id=client_id,
        node_uuids=msg_uuids,
        redis_client_async=redis_client_async,
    )


async def get_chat_history_for_chats(
    db: AsyncSession, chat_ids: list[int], get_msgs_in_index: bool = True
) -> list[uuid.UUID]:
    stmt = (
        select(models.ChatHistory)
        .filter(models.ChatHistory.chat_id.in_(chat_ids), models.ChatHistory.msg_in_index.is_(get_msgs_in_index))
        .order_by(models.ChatHistory.timestamp)
    )
    result = await db.execute(stmt)
    msgs = result.scalars().all()
    return [msg.msg_uuid for msg in msgs]


async def index_chat_history(
    db: AsyncSession,
    client_uuid: uuid.UUID,
    chat_id: int,
    chat_uuid: uuid.UUID,
    vector_id: int,
    chat_history: list[sch.APIMessage],
    callback_manager: CallbackManager,
    vector_store: BasePydanticVectorStore,
    embedding_model_info: sch.AzureModelInfo,
) -> None:
    chat = await get_chat_from_id(db=db, chat_id=chat_id)
    assert chat
    # Reindex old chats if not in the index
    if not chat.chat_in_index:
        old_chat_hist = []
        limit = constants.MAX_CHAT_HISTORY - len(chat_history)
        limited_chat_hist = await get_chat_history_limited(db=db, chat_id=chat_id, limit=limit)
        chat_hist_uuids = [str(msg.msg_uuid) for msg in chat_history]  # Check for no overlapping msgs
        for msg in limited_chat_hist:
            if msg.msg_in_index:
                continue
            if str(msg.msg_uuid) not in chat_hist_uuids:
                old_chat_hist.append(sch.APIMessage.from_orm(msg))
        chat_history = old_chat_hist + chat_history  # Put old chat hist first
    chat.chat_in_index = True
    await db.commit()
    # Add the chats to the vector database
    ai_catalog = AICatalog(callback_manager=callback_manager)
    vector_memory = VectorMemory.from_defaults(
        vector_store=vector_store,
        embed_model=ai_catalog.get_embedding_model(model_info=embedding_model_info),
    )
    logger.debug("Using the following chat_uuid %s", chat_uuid)
    logger.debug("Indexing the following chat history: %s", chat_history)
    for chat_msg in chat_history:
        idx_msg = copy.deepcopy(chat_msg)
        idx_msg.content = add_message_context(
            content=idx_msg.content,
            sql_query=idx_msg.sql_query,
            timestamp=idx_msg.timestamp,
            result_uuid=idx_msg.result_uuid,
            visual_json=idx_msg.visual_json,
        )
        # TODO: Update the exclude LLM and exclude embed for chat and client UUIDs
        await vector_memory.async_put(
            ChatMessage(content=idx_msg.content, timestamp=idx_msg.timestamp, role=idx_msg.role),
            {
                "chat_uuid": str(chat_uuid),
                "client_uuid": str(client_uuid),
                "vector_type": enums.VectorSourceType.CHAT.value,
            },
            str(idx_msg.msg_uuid),
        )


async def get_initial_prompt_for_result(db: AsyncSession, result_uuid: uuid.UUID):
    stmt = (
        select(models.ResultHistory.initial_prompt, models.ResultHistory.timestamp)
        .filter_by(result_uuid=result_uuid)
        .distinct()
        .order_by(models.ResultHistory.timestamp)
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_thumb_reaction_counts(db: AsyncSession, number_of_days: int = 7) -> Row:
    """Get the message thumb reaction"""
    # Filtering to just user to only get 1 count per message
    combined_stmt = (
        select(
            func.count().label("total_messages"),
            func.sum(case((models.ChatHistory.thumbs_up == False, 1), else_=0)).label("thumbs_down_count"),  # noqa
            func.sum(case((models.ChatHistory.thumbs_up == True, 1), else_=0)).label("thumbs_up_count"),  # noqa
        )
        .select_from(models.ChatHistory)
        .filter(
            models.ChatHistory.role == sch.MessageRole.USER.value,
            models.ChatHistory.timestamp >= (datetime.utcnow() - timedelta(days=number_of_days)),
        )
    )
    result = await db.execute(combined_stmt)
    return result.one()
