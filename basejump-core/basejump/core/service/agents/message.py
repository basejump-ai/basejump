import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Union
from zoneinfo import ZoneInfo

import aiohttp
from llama_index.core.llms import MessageRole
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils
from basejump.core.database.crud import crud_chat
from basejump.core.models import constants, enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


class MessageHandler:
    def __init__(
        self,
        prompt_metadata: Union[sch.PromptMetadataBase, sch.PromptMetadata],
        query_result: Optional[sch.MessageQueryResult] = None,
    ):
        self.prompt_metadata = prompt_metadata
        self.query_result = query_result or sch.MessageQueryResult()

    def create_message(
        self,
        role: MessageRole,
        content: str = "",
        msg_type: enums.MessageType = enums.MessageType.RESPONSE,
        msg_uuid: Optional[uuid.UUID] = None,
    ):
        if not msg_uuid:
            msg_uuid = uuid.uuid4()
        self.message = sch.Message(
            msg_uuid=msg_uuid,
            role=role,
            content=db_utils.remove_message_context(content=content),
            msg_type=msg_type,
            query_result=self.query_result,
            timestamp=datetime.now(ZoneInfo("UTC")).isoformat(),
        )


class ChatMessageHandler(MessageHandler):
    def __init__(
        self,
        prompt_metadata: Union[sch.PromptMetadataBase, sch.PromptMetadata],
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        query_result: Optional[sch.MessageQueryResult] = None,
        verbose: bool = False,
    ):
        super().__init__(prompt_metadata=prompt_metadata, query_result=query_result)
        self.chat_metadata = chat_metadata
        self.redis_client_async = redis_client_async
        self.verbose = verbose

    @property
    def api_message(self):
        return self.create_api_message()

    def _log_thought_message(self, content: str):
        thought = sch.ThoughtMessage(timestamp=datetime.now(ZoneInfo("UTC")), thought=content)
        self.chat_metadata.curr_thought_history.append(thought)

    def create_thought_message(self, content: str) -> None:
        super().create_message(role=sch.MessageRole.ASSISTANT, content=content, msg_type=enums.MessageType.THOUGHT)
        self._log_thought_message(content=content)

    async def create_message(  # type: ignore
        self,
        db: AsyncSession,
        role: MessageRole,
        content: str = "",
        msg_type: enums.MessageType = enums.MessageType.RESPONSE,
        msg_uuid: Optional[uuid.UUID] = None,
        initial_prompt: bool = False,
    ):
        super().create_message(role=role, content=content, msg_type=msg_type, msg_uuid=msg_uuid)
        if msg_type == enums.MessageType.THOUGHT:
            self._log_thought_message(content=content)
        if initial_prompt:
            # Save the user prompt
            self.chat_metadata.curr_chat_history.append(self.api_message)
            await crud_chat.save_message(
                db=db,
                message=self.api_message,
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                query_result=self.query_result,
            )
            # Save the assistant response placeholder
            super().create_message(role=MessageRole.ASSISTANT, msg_type=enums.MessageType.INIT)
            self.chat_metadata.curr_chat_history.append(self.api_message)
            await crud_chat.save_message(
                db=db,
                message=self.api_message,
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                query_result=self.query_result,
            )
        if self.chat_metadata.reset_parent_msg_uuid:
            # Sending an extra message if a new parent msg UUID needs to be reset
            # TODO: Create message here with blanks
            if self.verbose:
                logger.debug("Webhook message: %s", "Sending solution status to indicate AI has finalized reply")
            await self._send_solution_message(db=db)
            self.chat_metadata.parent_msg_uuid = self.message.msg_uuid
            self.chat_metadata.reset_parent_msg_uuid = False

    def format_message(self) -> str:
        self.api_message.timestamp = self.api_message.timestamp.isoformat()
        self.api_message.prompt_time = self.api_message.prompt_time.isoformat()
        if self.verbose:
            logger.debug("Here is the timestamp: %s", str(self.api_message.timestamp))
        return self.api_message.model_dump_json()

    async def save_message(self, message: sch.Message) -> None:
        # Add running chat history for the VectorMemory
        found_match = False
        for hist_message in self.chat_metadata.curr_chat_history:
            if hist_message.role == message.role and str(hist_message.parent_msg_uuid) == str(
                self.chat_metadata.parent_msg_uuid
            ):
                logger.debug("Found chat hist match for role: %s", hist_message.role)
                found_match = True
                # Update the message
                hist_message.content = message.content
                hist_message.msg_type = message.msg_type
                if message.query_result:
                    query_res_dict = self.process_query_result(query_result=message.query_result)
                    for key, value in query_res_dict.items():
                        setattr(hist_message, key, value)
        if not found_match:
            self.chat_metadata.curr_chat_history.append(self.api_message)
        # TODO: Use websockets so the DB doesn't have to be saved to until the user
        # disconnects

    async def send_api_message(self, send_solution: Optional[sch.SendSolution] = None):
        """Send messages to the API"""

        if not self.message:
            raise ValueError("Create a message first using create_message")
        if not self.chat_metadata:
            # Need a webhook url for anything to be sent
            return
        if self.verbose:
            logger.debug("Webhook message: %s", self.message.content)
        api_message = self.format_message()
        await self._send_api_message(api_message=api_message)
        # Make sure to send a solution message after the error message
        if send_solution or self.message.msg_type == enums.MessageType.ERROR:
            # NOTE: This is so the initial message from the endpoint has time to resolve before an error
            # is sent. This avoids the messages getting out of order.
            if self.message.msg_type == enums.MessageType.ERROR:
                await asyncio.sleep(1)
            if send_solution:
                await self._send_solution_message(db=send_solution.db)

    async def _send_api_message(self, api_message: str):
        if self.verbose:
            logger.debug("Webhook API message: %s", api_message)
        try:
            assert self.chat_metadata.webhook_url
            webhook_url = self.chat_metadata.webhook_url
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    headers={**self.chat_metadata.webhook_headers, "Content-Type": "application/json"},  # type: ignore
                    url=webhook_url,
                    data=api_message,
                ) as response:
                    if response.status != 200:
                        logger.warning("Webhook status response: %s", response.status)
                        logger.warning("Webhook status text: %s", response.text)
        except AssertionError:
            if self.verbose:
                logger.debug("No webhook URL found")
                logger.debug("Webhook header values %s", str(self.chat_metadata.webhook_headers))
            # If no webhook URL, then skip sending the API message
            pass

    def process_query_result(self, query_result: sch.MessageQueryResult) -> dict:
        if self.prompt_metadata.return_visual_json and query_result:
            if isinstance(query_result.visual_json, str):
                visual_json = json.loads(query_result.visual_json)
            else:
                visual_json = query_result.visual_json
        else:
            visual_json = None
        query_result = sch.MessageQueryResult(
            result_uuid=query_result.result_uuid if query_result.result_uuid else None,
            sql_query=query_result.sql_query,
            result_type=query_result.result_type,
            visual_result_uuid=(query_result.visual_result_uuid if query_result.visual_result_uuid else None),
            visual_json=visual_json,
            visual_explanation=query_result.visual_explanation,
        )
        query_result_dict = query_result.model_dump()
        return query_result_dict

    def create_api_message(self) -> sch.APIMessage:
        query_result_dict = self.process_query_result(query_result=self.query_result)
        if self.chat_metadata.semcache_response:
            verified_user_role = self.chat_metadata.semcache_response.verified_user_role
            can_verify = self.chat_metadata.semcache_response.can_verify
            verified_user_uuid = self.chat_metadata.semcache_response.verified_user_uuid
            verified = self.chat_metadata.semcache_response.verified
        else:
            verified_user_role = None
            can_verify = None
            verified_user_uuid = None
            verified = False
        # HACK: Need to use a constant instead: https://github.com/Basejump-AI/Basejump/issues/1441
        if self.message.content.strip() == "Reached max iterations.":
            raise Exception("Reached max iterations.")
        api_message = sch.APIMessage(
            # vars from ChatMessage
            role=self.message.role,
            msg_type=self.message.msg_type,
            # content=self.message.content,
            # TODO: Move this into be passed in the body instead
            # special characters in the content can cause issues if sent via header
            content=self.message.content,
            timestamp=self.message.timestamp,
            msg_uuid=self.message.msg_uuid,
            # vars from PromptMetadata
            prompt_uuid=self.prompt_metadata.prompt_uuid,
            initial_prompt=self.prompt_metadata.initial_prompt,
            prompt_time=self.prompt_metadata.prompt_time,
            parent_msg_uuid=self.chat_metadata.parent_msg_uuid,
            verified=verified,
            verified_user_role=verified_user_role,
            verified_user_uuid=verified_user_uuid,
            can_verify=can_verify,
            # vars from QueryResult
            **query_result_dict,
        )
        return api_message

    async def save_messages(self, db: AsyncSession):
        assert isinstance(self.prompt_metadata, sch.PromptMetadata)
        # TODO: Performance could possibly be improved to not update the vector DB table every time
        # for the index_created flg
        await crud_chat.index_chat_history(
            db=db,
            client_uuid=self.prompt_metadata.client_uuid,
            chat_id=self.chat_metadata.chat_id,
            chat_uuid=self.chat_metadata.chat_uuid,
            vector_id=self.chat_metadata.vector_id,
            chat_history=self.chat_metadata.curr_chat_history,
            callback_manager=self.prompt_metadata.callback_manager,
            vector_store=self.chat_metadata.vector_store,
            embedding_model_info=self.chat_metadata.embedding_model_info,
            verbose=self.verbose,
        )
        for api_message in self.chat_metadata.curr_chat_history:
            await crud_chat.save_message(
                db=db,
                message=api_message,
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                query_result=self.query_result,
                msg_in_index=True,
            )
        # Remove everything from current chat history
        self.chat_metadata.curr_chat_history = []
        # Remove extra chat history to prevent vector DB from getting too large
        msg_uuids = await crud_chat.get_chat_history_for_chats(db=db, chat_ids=[self.chat_metadata.chat_id])
        len_chats = len(msg_uuids)
        if len_chats > constants.MAX_CHAT_HISTORY:
            # Find the number of chat messages to remove
            chat_num_to_remove = len_chats - constants.MAX_CHAT_HISTORY
            await crud_chat.delete_chat_msgs_from_vector(
                db=db,
                client_id=self.prompt_metadata.client_id,
                msg_uuids=msg_uuids[:chat_num_to_remove],
                redis_client_async=self.redis_client_async,
            )

    async def _send_solution_message(self, db: AsyncSession):
        try:
            assert isinstance(self.prompt_metadata, sch.PromptMetadata)
        except AssertionError:
            raise AssertionError(
                "send_solution_message can only be used within the background task after helper_run_chat has been ran"
            )
        await self.save_messages(db=db)
        super().create_message(
            role=MessageRole.ASSISTANT,
            content="",
            msg_type=enums.MessageType.SOLUTION,
        )
        api_message = self.format_message()
        await self._send_api_message(api_message=api_message)
