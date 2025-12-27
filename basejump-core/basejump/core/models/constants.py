SQL_EXEC_TOOL_NM_PREFIX = "run_sql"
SQL_TABLES_TOOL_NM_PREFIX = "get_sql_tables"
VIS_TOOL_NM = "plot_tool"
INTERNAL_DOCS_TOOL_NM = "internal_docs_tool"
SQL_SUB_QUESTIONS = "simplify_user_prompt"
PUBSUB_STOPWORD = "STOP"
PUBSUB_DONEWORD = "DONE"
SQL_OPTION_1 = "Provide an answer to the user's original prompt using the results provided above from the SQL query"
SQL_OPTION_2_SUFFIX = "to ask a follow-up question to get other available relevant table information"
SQL_OPTION_3_SUFFIX = "to run another query in order to finish answering the user's question"
SQL_QUERY_TXT = " SQL QUERY:"
TIMESTAMP_TXT = " TIMESTAMP:"
VISUAL_RESULT_UUID = " RESULT_UUID FOR VISUAL:"
VISUAL_CONFIG = " VISUAL CONFIG OPTIONS USED:"
MAX_CHAT_HISTORY = 40
MESSAGE_TIMEOUT = 60 * 20
STILL_THINKING_TIME = 25
MAX_THOUGHT_TIME = 60
MSG_TIMED_OUT = "Message timed out. Please try again."
AI_RESULT_PREVIEW_CT = 10
MAX_ITERATIONS = 15
MAX_CHAT_HISTORY_DAYS = 365
# This distance is close enough that is likely only affects the SQL where clause
REDIS_SEMCACHE_SIMILAR_DISTANCE = 0.3
# This distance is close enough that it is assumed it's the exact same question
REDIS_SEMCACHE_EXACT_DISTANCE = 0.04
SEMCACHE_TIMESTAMP = "timestamp"
TEST_PROMPT = "give me a report of all clients"
REINDEXING_DB_ERROR_MSG = "Currently re-indexing database metadata. Please wait a few minutes and try again."
INDEX_DB_ERROR_MSG = "Currently indexing database metadata. Please wait a few minutes and try again."
NO_TABLES = """No tables found for this database connection. \
Please contact your administrator to review the database connection."""
NO_PERMITTED_TABLES = """No permitted tables found for this database connection. \
There are tables available, but none are permitted for your team. \
Please contact your administrator to provide access to tables."""
CONTENT_MGMT_POLICY = "prompt triggering Azure OpenAI's content management policy"
SQLALCHEMY_TIMEOUT = "reached, connection timed out"
THUMBS_UP_DESCR = """User reaction to the trustworthiness of the response from the AI. True means trusted, \
False (thumbs down) means incorrect, and no reaction is None."""
DB_ALIAS_NAME_DESC = """The display name for your database. \
Useful if you have multiple databases set up based on the same database."""
VECTOR_FILTERS: list = [
    {"type": "tag", "name": "name"},
    {"type": "tag", "name": "chat_uuid"},
    {"type": "tag", "name": "db_uuid"},
    {"type": "tag", "name": "client_uuid"},
    {"type": "tag", "name": "vector_type"},
]
UNRESOLVED_JINJA = "A connection used for this team does not have a jinja value provided for its schema. \
Ask your admin to update your database connections to include jinja values for all connections with jinjafied schemas."


def get_sql_tables_tool_nm(conn_id: int) -> str:
    return f"{SQL_TABLES_TOOL_NM_PREFIX}_{conn_id}"


def get_sub_questions_tool_nm(conn_id: int) -> str:
    return f"{SQL_SUB_QUESTIONS}_{conn_id}"


def get_sql_execution_tool_nm(conn_id: int) -> str:
    return f"{SQL_EXEC_TOOL_NM_PREFIX}_{conn_id}"


# Descriptions
USER_UUID_DSC = "Universally unique user ID."
TEAM_UUID_DSC = "Universally unique team ID."
CLIENT_UUID_DSC = "Universally unique client ID."
SUB_UUID_DSC = "Universally unique subscription ID."
CHAT_UUID_DSC = "Universally unique chat ID."
SAVED_RESULT_UUID_DSC = "Universally unique result ID for a saved generated data object."
SAVED_RESULT_UUID_DSC = "Universally unique saved result ID for generated data objects."
CHAT_NAME_DSC = "The name to give the chat."
CHAT_DESCRIPTION_DSC = "The chat description."
CONN_UUID_DSC = "Universally unique connection ID."
DB_UUID_DSC = "Universally unique database ID."
TBL_UUID_DSC = "Universally unique table ID."
COL_UUID_DSC = "Universally unique column ID."
RESULT_UUID_DSC = "Universally unique result ID for a generated data object."
VISUAL_RESULT_UUID_DSC = "Universally unique visual result ID for a generated visualization."
MSG_UUID_DSC = "Universally unique message ID."
PROMPT_UUID_DSC = "Universally unique prompt ID."
PARENT_MSG_UUID_DSC = "Universally unique parent message ID. This is used to AI thoughts and the final reply together."
SUB_PLAN_UUID_DSC = "Universally unique subscription plan ID."
PAGE_NUMBER_DSC = "The current page number to return."
ROWS_PER_PAGE_DSC = "The number of messages to return per page."
TOTAL_ROW_CT_DSC = """This field can be ignored. It is meant to pass the table row count between calls to avoid \
regenerating every time when using next_page_link and prev_page_link"""
TEAM_NM_DESC = "Your team name here"
TEAM_DESC = "A description of the team. This is provided to the AI as context."
ATHENA_STAGING_DIR_NAME = "s3_staging_dir"
