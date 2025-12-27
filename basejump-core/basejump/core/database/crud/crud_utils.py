import json

import sqlalchemy as sa
import tiktoken
from basejump.core.database.db_connect import ConnectDB
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.models.models import Base, DBParams
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    TokenCountingHandler,
)
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession


async def get_next_val(db: AsyncSession, full_table_nm: str, column_nm: str):
    """Get the next value in a sequence to avoid having to commit and refresh the table"""
    table_seq_base = await db.execute(sa.text(f"SELECT pg_get_serial_sequence('{full_table_nm}', '{column_nm}');"))
    table_seq = table_seq_base.scalar()
    if not table_seq:
        table_seq = f"{full_table_nm}_{column_nm}_seq"
    next_val_base = await db.execute(sa.text(f"SELECT nextval('{table_seq}')"))
    return next_val_base.scalar()


def create_callback_mgrs() -> sch.CallbackMgrs:
    """
    Set up callback manager scoped to this specific conversation

    Notes
    -----
    Do not move this out of this function to the app level
    """
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model(enums.AIModelSchema.GPT4o.value).encode)
    callback_manager = CallbackManager([llama_debug, token_counter])
    callback_mgrs = sch.CallbackMgrs(
        token_counter=token_counter, callback_manager=callback_manager, llama_debug=llama_debug
    )
    return callback_mgrs


def update_model(schema: BaseModel, db_model: Base):
    for key, value in schema.model_dump(exclude_unset=True).items():
        if isinstance(value, dict):
            value = json.dumps(value)
        if value is not None:
            setattr(db_model, key, value)
    return db_model


def helper_decrypt_db(database: DBParams) -> sch.GetDBParams:
    value = sch.GetDBParamsBytes.model_validate(database)
    value_dict = value.model_dump()
    db_uuid = value_dict["db_uuid"]
    del value_dict["db_uuid"]
    return_dict = ConnectDB.decrypt_db(value_dict)
    # BC v0.27.0: Schemas used to be null so handling that case
    if not return_dict.get("schemas"):
        return_dict["schemas"] = []
    if return_dict.get("connections"):
        if not return_dict["connections"]["schemas"]:
            return_dict["connections"]["schemas"] = []
    db_params = sch.GetDBParams(**return_dict, db_uuid=db_uuid)
    return db_params
