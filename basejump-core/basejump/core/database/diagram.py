"""Create ERD diagrams from the database"""

import math
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat, crud_connection
from basejump.core.database.format_response import JSONResponseFormatter
from basejump.core.database.vector_utils import get_table_info_from_vector_db
from basejump.core.models import enums, prompts
from basejump.core.models import pydantic_ai_formats as fmt
from basejump.core.models import schemas as sch
from basejump.core.service.agents.mermaid import MermaidAgent
from basejump.core.service.base import AgentSetup
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

logger = set_logging(handler_option="stream", name=__name__)

MERMAID_PAGE_LIMIT = 25
VECTOR_METADATA_TBL_INFO_COL_NM = "db_table_info"


class MermaidAgentManager:
    def __init__(
        self,
        db: AsyncSession,
        mermaid_agent: MermaidAgent,
        index_name: str,
        tbl_uuids: list[uuid.UUID],
        client_user: sch.ClientUserInfo,
        vector_uuid: uuid.UUID,
        small_model_info: sch.ModelInfo,
        large_model_info: sch.ModelInfo,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
    ):
        self.db = db
        self.index_name = index_name
        self.tbl_uuids = tbl_uuids
        self.client_user = client_user
        self.vector_uuid = vector_uuid
        self.large_model_info = large_model_info
        self.small_model_info = small_model_info
        self.redis_client_async = redis_client_async
        self.sql_engine = sql_engine
        self.mermaid_agent = mermaid_agent

    async def _update_agent_prompt(self, prompt: str) -> None:
        # Update the agent prompts
        prompt_id, prompt_uuid = await crud_chat.create_prompt_history(
            db=self.db, client_id=self.client_user.client_id, llm_type=enums.LLMType.MERMAID_AGENT
        )

        prompt_metadata_base = sch.PromptMetadataBase(
            initial_prompt=prompt,
            user_id=self.client_user.user_id,
            user_uuid=self.client_user.user_uuid,
            client_uuid=self.client_user.client_uuid,
            client_id=self.client_user.client_id,
            user_role=self.client_user.user_role,
            prompt_uuid=prompt_uuid,
            prompt_id=prompt_id,
            llm_type=enums.LLMType.MERMAID_AGENT,
            prompt_time=datetime.now(),
        )
        agent_setup = AgentSetup.load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
        self.mermaid_agent.prompt_metadata = agent_setup.prompt_metadata

    async def _paginate_mermaid_code(self, start: int) -> str:
        # Set up mermaid agent
        db_table_info = await get_table_info_from_vector_db(
            index_name=self.index_name,
            tbl_uuids=self.tbl_uuids,
            start=start,
            offset=MERMAID_PAGE_LIMIT,
            redis_client_async=self.redis_client_async,
        )
        prompt = prompts.MERMAIDJS_PROMPT.format(table_schemas=db_table_info)
        logger.debug("here is the prompt %s", prompt)
        await self._update_agent_prompt(prompt=prompt)
        # Retrieve the diagram
        diagram_code = await self.mermaid_agent.retrieve_mermaidjs_diagram()
        return diagram_code

    async def paginate_mermaid_code(self) -> str:
        """When there are many tables, the responses need to be compiled by requesting \
the AI to process them in chunks"""
        num_of_tables = len(self.tbl_uuids)
        pages = math.ceil(num_of_tables / MERMAID_PAGE_LIMIT)
        diagram_codes = []
        for idx in range(pages):
            logger.info(f"Processing Mermaid diagram chunk {idx+1} of {pages} pages")
            diagram_code = await self._paginate_mermaid_code(start=idx * MERMAID_PAGE_LIMIT)
            diagram_codes.append(diagram_code)
        return await self.process_paginated_diagram_code(diagram_codes=diagram_codes)

    async def process_paginated_diagram_code(self, diagram_codes: list):
        mermaid_body = []
        mermaid_appendix = []
        start_idx = None
        for diagram_code in diagram_codes:
            code_chunks = diagram_code.split("\n")
            for idx, chunk in enumerate(code_chunks):
                if "erDiagram" in chunk:
                    start_idx = idx
                if "--o{" in chunk:
                    if not start_idx:
                        raise Exception("Needs to have erDiagram")
                    # Start at 2 since the first 2 include mermaid + erDiagram
                    mermaid_body += code_chunks[start_idx:idx]
                    mermaid_appendix += code_chunks[idx:]
                    break
        # Combine mermaid diagram back together
        final_diagram = (
            "erDiagram"
            + "\n".join(mermaid_body).replace("erDiagram", "")
            + "\n".join(mermaid_appendix).replace("```", "")
        )
        if len(final_diagram) > 15000:
            # ~15K characters is roughly 4096 tokens - if output limit is reached then return
            # what we have currently without extracting
            return final_diagram
        else:
            format_json_response = JSONResponseFormatter(
                response=final_diagram,
                pydantic_format=fmt.MermaidJSFormat,
                llm=self.mermaid_agent.agent_llm,
                small_model_info=self.small_model_info,
            )
            extract = await format_json_response.format()
            return extract.mermaidjs_code

    async def create_erd_diagram(self) -> str:
        try:
            diagram_code = await self.paginate_mermaid_code()
        except Exception as e:
            logger.error(str(e))
            raise e
        # Update the database with the diagram
        try:
            update_vector = sch.UpdateVector(
                vector_metadata={VECTOR_METADATA_TBL_INFO_COL_NM: diagram_code},
                timestamp=datetime.now(ZoneInfo("UTC")),
            )
            logger.debug("Here is the UpdateVector: %s", update_vector)
            logger.debug("Here is the client ID: %s", str(self.client_user.client_id))

            await crud_connection.update_vector_connection(
                db=self.db, update_vector=update_vector, vector_uuid=self.vector_uuid
            )
        except Exception as e:
            logger.error(str(e))
            raise e
        return diagram_code
