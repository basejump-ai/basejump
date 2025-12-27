"""Store prompts to use with the AI"""

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants, enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

DB_METADATA_PROMPT = (
    "Here are the SQL tables relevant to your following inquiry (in order of relevance): {inquiry}\n"
    "When creating a SQL query, only use the tables listed below:\n"
    "{schema}\n\n"
    "Follow these rules when creating SQL queries:\n"
    "- Given an input question, first create a syntactically correct {db_type} "
    "query to run, then look at the results of the query and return the answer. \n"
    "- You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n"
    "- Never query for all the columns from a specific table, only ask for a "
    "few relevant columns given the question.\n"
    "- Pay attention to use only the column names that you can see in the schema "
    "description. \n"
    "- Be careful to not query for columns that do not exist. \n"
    "- Pay attention to which column is in which table. \n"
    "- Don't use limit unless the user asks you only for a certain number of results \n"
    f"- Keep in mind you only get to preview the first {constants.AI_RESULT_PREVIEW_CT} rows of any query result "
    "but the user will see all rows.\n"
    "- Don't use backslashes for line breaks or continuation in SQL strings"
    "- If multiple queries are needed to answer the question, use CTEs for the intermediate SQL query steps. \n"
    "- If you need to filter a column by a particular string, ensure it is a distinct value for the column "
    "you are filtering based on the tables listed above. If it is not one of the distinct values, use a LIKE "
    "operator as a fuzzy match and confirm what the correct value should be. Confirm with "
    "the user the updated string value you would like to filter by before running the SQL query. \n"
    "- Qualify column names with the table name when needed.\n\n"
    "Now do one of the following: \n"
    "- Option 1: Ask the user a clarifying question to have better context for your SQL query if you \
feel certain filters aren't clear based on their prompt.\n"
    "- Option 2: Run your SQL query using the following tool: {run_sql_query_tool}"
)

MERMAIDJS_SYSTEM_PROMPT = """\
You are an agent that creates mermaidjs entity relationship diagrams (ERD). Always validate your \
mermaid code using the provided tool before answering the user. Replace all periods in table names \
with underscores since the diagram won't parse correctly if periods are used. Make sure to remember \
to pass your mermaid code as an argument to the tool for validation.

Here is an example of a diagram:
erDiagram
    JOBS {
        INTEGER job_id
        VARCHAR job_title
        DECIMAL min_salary
        DECIMAL max_salary
    }
    DEPARTMENTS {
        INTEGER department_id
        VARCHAR department_name
        INTEGER location_id
    }
    DEPENDENTS {
        INTEGER dependent_id
        VARCHAR first_name
        VARCHAR last_name
        VARCHAR relationship
        INTEGER employee_id
    }
    EMPLOYEES {
        INTEGER employee_id
        VARCHAR first_name
        VARCHAR last_name
        VARCHAR email
        VARCHAR phone_number
        DATE hire_date
        INTEGER job_id
        DECIMAL salary
        INTEGER manager_id
        INTEGER department_id
    }
    LOCATIONS {
        INTEGER location_id
        VARCHAR street_address
        VARCHAR postal_code
        VARCHAR city
        VARCHAR state_province
    }
    DEPARTMENTS ||--o{ LOCATIONS : "located at"
    DEPENDENTS ||--o{ EMPLOYEES : "dependent of"
    EMPLOYEES ||--o{ JOBS : "has job"
    EMPLOYEES ||--o{ DEPARTMENTS : "works in"
    EMPLOYEES ||--o{ EMPLOYEES : "managed by"

Note that you do not need to specify the number of digits, characters, etc...

For example, instead of DECIMAL(10), just put DECIMAL. Only specify the type for each table without
additional information for the table column.
"""
MERMAIDJS_PROMPT = """
Create a mermaidjs diagram only using the following table schemas:

{table_schemas}
"""


def sql_result_prompt_basic(query_result: sch.QueryResult):
    return f"""Here are the results of your SQL query up to {query_result.ai_preview_row_ct} \
rows of {str(query_result.num_rows)} rows:\n
{str(query_result.query_result)}\n\n

The query results will be displayed to the user after your comment. Respond \
to the user letting them know about the result. For example, if the user asked, "I want to see all records" \
then you would respond "Here are the records you requested" or "Here is a report of all the records" or \
"Here is a dataset of all the records". Do not mention anything about the results being displayed to the user. \
They will see it in the chat window. Do not list the {query_result.ai_preview_row_ct} \
rows of output in your response. \
Talk as if you are handing them a dataset in person."""


ZERO_ROW_PROMPT = """
Do one of the following:
- Option 1: Search the database for similar values for a filter you suspect to be the cause of zero rows. \
If there are many similar values, ask they user which one they want or if they want them all.
- Option 2: Respond to the user letting them know that no rows were returned. \
Let the user know what filters there are in the query and the tables that were used in the query. \
Use this format to inform the user of the filters:
Tables:
- Table: <Table name here>

Filters:
- Column: <Column name here>, Filter: <filter description>
Ask the user if they would like you to search for similar filter values or if they would like you \
to change any of them. Also advise the user to check the spelling. \
Also let them know they can also review the tables you used in the 'Explore' page and ask a question \
specifying columns and/or tables. Finally, they then can try re-phrasing the prompt."""


# TODO: Let the AI know how many attempts it has remaining
def get_sql_result_prompt(conn_id: int, query_result: sch.QueryResult):
    sql_tbl_tool_nm = constants.get_sql_tables_tool_nm(conn_id=conn_id)
    sql_exec_tool_nm = constants.get_sql_execution_tool_nm(conn_id=conn_id)
    if query_result.num_rows == 0:
        logger.info("Query returned no rows")
        return """That query returned 0 rows.\n""" + ZERO_ROW_PROMPT
    SQL_ACTION_OPTIONS = f"""\
Do one of the following:
Option 1. {constants.SQL_OPTION_1}
Option 2. Use the {sql_tbl_tool_nm} {constants.SQL_OPTION_2_SUFFIX}
Option 3. Use the {sql_exec_tool_nm} {constants.SQL_OPTION_3_SUFFIX}
Option 4. Use the {constants.VIS_TOOL_NM} to create a chart based on the users request.\
The result_uuid to do this is {query_result.result_uuid}"""
    if query_result.result_type == enums.ResultType.DATASET:
        query_result_str = f"""
Here are the results of your SQL query up to {query_result.ai_preview_row_ct} \
rows of {str(query_result.num_rows)} rows:\n
{str(query_result.query_result)}\n\n
{SQL_ACTION_OPTIONS}

If option 1 is selected, follow these instructions: \
The query results will be displayed to the user after your comment. Respond \
to the user letting them know about the result. For example, if the user asked, "I want to see all records" \
then you would respond "Here are the records you requested" or "Here is a report of all the records" or \
"Here is a dataset of all the records". Do not mention anything about the results being displayed to the user. \
They will see it in the chat window. Do not list the {query_result.ai_preview_row_ct} \
rows of output in your response. \
Talk as if you are handing them a dataset in person."""
    else:
        query_result_str = f"""\
Here are the results of your SQL query for the first row of {str(query_result.num_rows)} rows:\n
{str(query_result.query_result[0])}\n
{SQL_ACTION_OPTIONS}
"""
    return query_result_str


COMPARE_QUERIES_PROMPT = """The following two queries have been labelled as {label.value}. This means \
{label.description}.\n
Explain in more detail why the following queries were given the {label.value} label:\n\n
Compared from query with title of "{source_result.result_title}" and query: {source_result.sql_query}\n\n
Compared against query with title of "{target_result.result_title}" and query: {target_result.sql_query}\n
Keep your answer brief with a maximum of 5 sentences. In your response, don't repeat the entire SQL statements\
provided in this prompt. When referring to queries, please use the query titles instead of referencing their order.\
For example, say '{source_result.result_title} query' instead of 'the first query'
"""


NO_DB_ACCESS_PROMPT = """You don't have access to a database. \
Do not offer SQL suggestions based on guesses of the database schema. \
If the answer must be answered by a database, then inform the user that the user's designated \
admin hasn't given you access to any company database yet and \
you can't answer the question without database access. Otherwise, if the answer can be given \
without database access (such as answers based on prior chat messages), then answer the user's question. \
Here is the user's question (address the user directly when responding): {prompt}"""
