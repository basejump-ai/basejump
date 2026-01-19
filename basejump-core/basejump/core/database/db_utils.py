"""Functions related to relational databases"""

import re
import string
import uuid
from datetime import datetime
from typing import Optional, Union

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants
from basejump.core.models import schemas as sch
from basejump.core.models.models import Base

logger = set_logging(handler_option="stream", name=__name__)


def remove_message_context(content: str) -> str:
    # Timestamp is added first, so if I split on that then I can separate out the correct context
    if constants.TIMESTAMP_TXT in content:
        content = content.split(constants.TIMESTAMP_TXT)[0]
    return content


def _update_visual_info(visual_info: str, visual_dict: dict, key: str) -> str:
    """Internal function to update visualization info metadata

    Parameters
    ----------
    visual_info
        This is a string of the visual information collected so far
    visual_dict
        A dictionary from visual_json
    key
        The dictionary key to retrieve
    """
    value = visual_dict.get(key)
    if value:
        visual_info += f" {key} = " + str(value)
    return visual_info


def extract_visual_info(visual_json: dict) -> str:
    """Take the visual plotly dictionary and parse it"""
    visual_info = ""
    try:
        # TODO: Determine when there would be more than one item in the list and then update this function
        data_dict = visual_json.get("data")[0]  # type: ignore
        for option in ["yaxis", "xaxis"]:
            visual_info = _update_visual_info(visual_info=visual_info, visual_dict=visual_json, key=option)
        if data_dict:
            for data_option in ["type", "y", "x", "orientation"]:
                visual_info = _update_visual_info(visual_info=visual_info, visual_dict=data_dict, key=data_option)
    except Exception as e:
        logger.warning("Error parsing visual_json: %s", str(e))
    return visual_info


def add_message_context(
    content: str,
    timestamp: Optional[Union[datetime, str]] = None,
    sql_query: Optional[str] = None,
    result_uuid: Optional[uuid.UUID] = None,
    visual_json: Optional[dict] = None,
) -> str:
    # HACK: Adding the SQL query to the response message to improve SQL query recall.
    # There is probably a more elegant way to do this.
    # NOTE: It's important that the timestamp is first since that is what is used to remove the chat metadata
    # NOTE: If this is updated, make sure to update the system prompt as well
    content += f"{constants.TIMESTAMP_TXT} {str(timestamp)}"
    if sql_query:
        content += f"{constants.SQL_QUERY_TXT} {sql_query}"
    if result_uuid:
        content += f"{constants.VISUAL_RESULT_UUID} {str(result_uuid)}"
    if visual_json:
        assert result_uuid, "Visual JSON needs to be associated with a result"
        content += f"{constants.VISUAL_CONFIG}"
        visual_info = extract_visual_info(visual_json=visual_json)
        content += visual_info

    return content


def process_foreign_key_definition(f_constraint_def: str) -> dict:
    # Define the regex pattern
    pattern = r"FOREIGN KEY \(([^)]+)\) REFERENCES ([^\.]+)\.([^\(]+)\(([^)]+)\)"

    # Use re.search to find the match
    match = re.search(pattern, f_constraint_def)

    if match:
        # Extract the groups
        foreign_key_column = match.group(1)
        referred_table = f"{match.group(2)}.{match.group(3)}"
        referred_column = match.group(4)

    return {
        "constrained_columns": foreign_key_column,
        "referred_table": referred_table,
        "referred_column": referred_column,
    }


def get_column_str(column: sch.DBColumn):
    if column.schema_name:
        return f"{column.schema_name}.{column.table_name}.{column.column_name}"
    return f"{column.table_name}.{column.column_name}"


def get_table_name_from_column(column: sch.DBColumn):
    if column.schema_name:
        return f"{column.schema_name}.{column.table_name}"
    else:
        return column.table_name


def get_table_name(table_name: str):
    return table_name.split(".")[1] if len(table_name.split(".")) > 1 else table_name


def get_table_schema(table_name: str):
    return table_name.split(".")[0] if len(table_name.split(".")) > 1 else None


def fuzzify_filter_value(value):
    # TODO: Handle queries where the AI is using Regex
    logger.debug("fuzzifying value: %s", value)
    new_value = re.sub(f"[{re.escape(string.punctuation)}]", " ", value.lower()).strip().replace(" ", "%")
    final_value = f"%{new_value}%"
    logger.debug("fuzzified value: %s", final_value)
    return final_value


def get_query_column_values(query_result: sch.QueryResultDF) -> list:
    """Get a list of values based on a column"""
    try:
        return query_result.output_df.iloc[:, 0].to_list()
    except Exception as e:
        logger.error("Issue with getting dataframe values: %s", str(e))
        return []


async def process_db_tables(
    tables: list[sch.GetSQLTable],
    exclude_ignored_columns: bool = True,
    exclude_ignored_tables: bool = True,
    include_db_table: bool = False,
) -> list[sch.SQLTable]:
    tables_base = [
        sch.SQLTable(
            table_name=get_table_name(table_name=table.table_name),
            table_schema=get_table_schema(table_name=table.table_name),
            full_table_name=table.table_name,
            context_str=table.context,
            tbl_uuid=str(table.tbl_uuid),
            columns=[
                sch.SQLTableColumn(**column.dict())
                for column in table.columns
                if not column.ignore or not exclude_ignored_columns
            ],
            ignore=table.ignore,
            primary_keys=table.primary_keys,
        )
        for table in tables
        if not table.ignore or not exclude_ignored_tables
    ]
    return tables_base


def get_table_schemas() -> list:
    return list(set([table.split(".")[0] for table in Base.metadata.tables.keys() if len(table.split(".")) > 1]))


def get_table_names() -> list[tuple]:
    return list(
        set(
            [
                ((table.split(".")[0], table.split(".")[1]) if len(table.split(".")) > 1 else (None, table))
                for table in Base.metadata.tables.keys()
            ]
        )
    )
