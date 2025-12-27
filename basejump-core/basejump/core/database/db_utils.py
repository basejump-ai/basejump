"""Functions related to relational databases"""

import re
import string
import uuid
from datetime import datetime
from typing import Optional, Type, Union

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from sqlglot import diff
from sqlglot import errors as sqlglot_errors
from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import DialectType
from sqlglot.diff import Keep, Move
from sqlglot.expressions import Expression
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

logger = set_logging(handler_option="stream", name=__name__)


# NOTE: These are string expressions only
STRING_EXPRESSIONS_OPS = [exp.In, exp.Is]
STRING_EXPRESSION_OPS = [exp.EQ, exp.NEQ, exp.Like, exp.SimilarTo, exp.RegexpLike]
ALL_STRING_EXPRESSION_OPS = STRING_EXPRESSIONS_OPS + STRING_EXPRESSION_OPS
NON_STRING_EXPRESSION_OPS = [exp.Between]


def standardize_aliases(tree):
    """
    Parameters
    -----------
    tree
        The sqlglot parsed tree of a sql query string
    """
    olda = []
    oldt = []
    for a in tree.find_all(exp.Alias):
        olda.append(a.alias)

    for t in tree.find_all(exp.TableAlias):
        oldt.append(t.this.this)

    for i in tree.find_all(exp.Identifier):
        if i.this in olda:
            new = i.args
            new["this"] = "alias"
            i.replace(exp.Identifier(**new))
        if i.this in oldt:
            new = i.args
            new["this"] = "tblalias"
            i.replace(exp.Identifier(**new))
    return tree


# TODO: Update all inputs to compare SQL queries to have the dialect derived from the connection params
# using the result_conn_id
# GHI: https://github.com/Basejump-AI/Basejump/issues/1301
def compare_sql_queries(
    sql_source: str, sql_target: str, dialect: Optional[DialectType] = None
) -> enums.SQLSimilarityLabel:
    """Compares two SQL queries to each other and gives them a label"""

    # TODO: Use the sqlglot optimize function on the queries to produce a more efficient and accurate diff
    # This will also help the AI compare between them
    std_source = standardize_aliases(parse_one(sql_source, dialect=dialect))
    std_target = standardize_aliases(parse_one(sql_target, dialect=dialect))
    # Compare the queries
    sql_diffs = diff(std_source, std_target)
    if all([True if isinstance(item, Keep) else False for item in sql_diffs]):
        return enums.SQLSimilarityLabel.IDENTICAL

    if all([True if isinstance(item, Keep) or isinstance(item, Move) else False for item in sql_diffs]):
        return enums.SQLSimilarityLabel.EQUIVALENT

    # Find the tables
    parsed_source = parse_one(sql_source, dialect=dialect)
    parsed_target = parse_one(sql_target, dialect=dialect)
    source_tables = {table.name for table in parsed_source.find_all(exp.Table)}
    target_tables = {table.name for table in parsed_target.find_all(exp.Table)}

    if source_tables & target_tables:
        # If there are intersecting tables then they are similar
        return enums.SQLSimilarityLabel.SIMILAR

    return enums.SQLSimilarityLabel.DIFFERENT


def remove_where_clauses(ast):
    """Remove where clauses from a query"""
    # Traverse the AST to find subqueries
    for node in ast.find_all(exp.Select):
        # Check if the subquery has a WHERE clause
        if node.args.get("where"):
            # Remove the WHERE clause
            del node.args["where"]


def get_full_table_name(query, table_name):
    # Escape special regex characters in the table name
    escaped_table = re.escape(table_name)

    # Regex pattern to match schema.table_name after whole words FROM or INNER JOIN
    pattern = r"\b(?:FROM|INNER JOIN)\s+(\w+\." + escaped_table + r")\b"

    match = re.search(pattern, query, re.IGNORECASE)

    return match.group(1) if match else None


def remove_jinjafied_schemas(query: str, schemas: list[sch.DBSchema], dialect: Optional[DialectType]) -> str:
    """Replace tables with schemas that are jinjafied with just the table name"""
    # Get tables
    parsed_query = parse_one(query, dialect=dialect)
    tables = {table.name for table in parsed_query.find_all(exp.Table)}
    # Get full table names
    tables_to_update = {}
    for table in tables:
        full_table_name = get_full_table_name(query=query, table_name=table)
        add_table = False
        split_tbl_nm = full_table_name.split(".")
        if len(split_tbl_nm) == 1:
            continue
        for schema in schemas:
            if schema.jinja_values:
                # Get the jinja table name without jinja
                name_no_jinja = [item for item in re.split(r"\{\{.*?\}\}", schema.schema_nm) if item]
                for item in name_no_jinja:
                    if item in full_table_name:
                        add_table = True
                    else:
                        add_table = False
        # Remove the jinjafied portion of the value
        tbl_schema = split_tbl_nm[0]
        re_pattern = re.sub(r"\{\{.*?\}\}", "(.+)", schema.schema_nm)
        if not bool(re.match(f"^{re_pattern}$", tbl_schema)):
            logger.debug("Pattern did not match. Setting add_table to False.")
            add_table = False
        if add_table:
            tables_to_update[full_table_name] = table
    # Replace schemas if jinjafied
    for key, value in tables_to_update.items():
        logger.debug(f"Pattern match found. Replacing {key} with {value}")
        query = re.sub(key, value, query)
    return query


def compare_sql_queries_no_where_clause(
    query1, query2, dialect: Optional[DialectType], schemas: Optional[list[sch.DBSchema]] = None
) -> Optional[enums.SQLSimilarityLabel]:
    """Compare everything except the where clauses"""
    try:
        if schemas:
            logger.debug("Schemas found - checking if schemas need to be replaced.")
            query1 = remove_jinjafied_schemas(query=query1, schemas=schemas, dialect=dialect)
            query2 = remove_jinjafied_schemas(query=query2, schemas=schemas, dialect=dialect)
        ast1 = parse_one(query1, dialect=dialect)
        ast2 = parse_one(query2, dialect=dialect)
        for ast in [ast1, ast2]:
            remove_where_clauses(ast=ast)
        query1 = str(ast1)
        query2 = str(ast2)
        logger.debug("query1: %s", query1)
        logger.debug("query1: %s", query2)
        return compare_sql_queries(query1, query2)
    except (errors.SQLParseError, sqlglot_errors.ParseError) as e:
        logger.warning("SQLglot failed parsing: %s", str(e))
        return None


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


def check_aggregate(item) -> bool:
    agg_types = [exp.Max, exp.Min, exp.Sum, exp.Count, exp.Avg, exp.Stddev, exp.Variance]
    for agg in agg_types:
        if isinstance(item, agg):
            return True
    return False


def qualify_column_names(
    column: exp.Column, scope: Scope, depth=0, columns: Optional[list] = None
) -> list[sch.DBColumn]:
    """Finds the original source table for all columns and formats the column names to
    <schema>.<table_name>.<column_name>

    Parameters
    ----------
    column
        A column AST object
    scope
        A scope AST object

    Returns
    -------
    str
        Returns a fully qualified column name
    """
    # Search all columns for a matching name
    if not columns:
        columns = []
    column_scopes = []
    if depth > 0:
        for projection in scope.find(exp.Select).expressions:
            if projection.alias == column.name:
                if isinstance(projection.this, exp.Column):
                    column_scopes.append((projection.this, scope))
                elif check_aggregate(projection.this):
                    # Get all the columns
                    agg_cols = projection.this.find_all(exp.Column)
                    for agg_col in agg_cols:
                        column_scopes.append((agg_col, scope))
                break
    # Get all table sources
    if not column_scopes:
        column_scopes = [(column, scope)]
    for column, scope in column_scopes:
        for alias, (node, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table):
                if column.table in [source.name, source.alias]:
                    # TODO: Columns in the database aren't stored with their source schema if they are in the default
                    col_filters = get_column_filters(column=column)
                    # Check if the column is being cast
                    cast_type = get_column_cast(column=column)
                    column_w_func = get_column_func(column=column)
                    db_col = sch.DBColumn(
                        column_name=column.name,
                        table_name=source.name,
                        schema_name=source.db,
                        filters=col_filters,
                        cast_type=cast_type,
                        column_w_func=column_w_func,
                    )
                    columns.append(db_col)
            elif isinstance(source, Scope):
                if alias != column.table:
                    continue
                return qualify_column_names(column, source, depth + 1, columns)
    return columns


def check_for_star(sql_query: str, dialect: Optional[DialectType]) -> Optional[str]:
    """The SQLglot qualify method fails if any asterisks are used to select columns,
    so this query is used to check for any star and return a string if there is one
    guiding the LLM to not use asterisks to select columns.
    """
    ast = parse_one(sql_query, dialect=dialect)
    for select in ast.find_all(exp.Select):
        if select.is_star:
            logger.warning("A star character is being used in the SQL query by the AI")
            raise errors.StarQueryError
    return None


def consolidate_columns(columns: list[sch.DBColumn]) -> list[sch.DBColumn]:
    """Combine filters based on column names"""
    # TODO: Is this function necessary?
    consolidated_list = [columns[0]]
    for column in columns:
        match = False
        for col in consolidated_list:
            if (
                column.column_name == col.column_name
                and column.table_name == col.table_name
                and column.schema_name == col.schema_name
            ):
                match = True
                for filter_val in col.filters:
                    if filter_val not in column.filters:
                        column.filters.append(filter_val)
        if not match:
            consolidated_list.append(column)
    return consolidated_list


def quote_identifiers(sql: str, dialect: DialectType) -> str:
    def _quote_identifiers(node):
        if isinstance(node, exp.Identifier):
            node.set("quoted", True)
        return node

    return parse_one(sql, dialect=dialect).transform(_quote_identifiers).sql(dialect=dialect)


def unquote_identifiers(sql: str, dialect: DialectType) -> str:
    def _unquote_identifiers(node):
        if isinstance(node, exp.Identifier):
            node.set("quoted", False)
        return node

    return parse_one(sql, dialect=dialect).transform(_unquote_identifiers).sql(dialect=dialect)


def qualify_names(sql_query: str, dialect: DialectType) -> exp.Expression:
    logger.info("Getting fully qualified column names")
    logger.info("Dialect: %s", dialect)
    check_for_star(sql_query=sql_query, dialect=dialect)
    # quoting columns for case sensitivity in qualify because qualify lowercases everything
    sql_query = quote_identifiers(sql=sql_query, dialect=dialect)
    ast = parse_one(sql_query, dialect=dialect)
    try:
        qualified_ast = qualify(ast)
    except Exception as e:
        warn_msg = "Unable to qualify AST"
        logger.warning(warn_msg)
        logger.warning("Here is the error: %s", str(e))
        raise errors.SQLParseError(warn_msg) from e
    return qualified_ast


def get_fully_qualified_col_names(
    sql_query: str, dialect: DialectType, ancestor_to_filter: Optional[Type[exp.Expression]] = None
) -> list[sch.DBColumn]:
    """Get the fully qualified column names in the
    format of <schema name>.<table name>.<column name>

    Parameters
    ----------
    sql_query
        The SQL query string
    ancestor_to_filter
        Filter the columns if they have a certain ancestor type
    """
    try:
        qualified_ast = qualify_names(sql_query=sql_query, dialect=dialect)
    except errors.StarQueryError:
        logger.warning("A star character is being used in the SQL query by the AI")
        raise errors.StarQueryError
    root = build_scope(qualified_ast)

    if not root:
        warn_msg = "No root available from building scope"
        logger.warning(warn_msg)
        raise errors.SQLParseError(warn_msg)
    columns = []
    for scope in root.traverse():
        for column in scope.columns:
            if ancestor_to_filter:
                is_descendant = column.find_ancestor(ancestor_to_filter)
                if not is_descendant:
                    continue
            qualified_cols = qualify_column_names(column, scope)

            if qualified_cols is None:
                warn_msg = "Unable to qualify AST columns"
                logger.warning(warn_msg)
                raise errors.SQLParseError(warn_msg)
            columns.extend(qualified_cols)
    consldt_cols = []
    if columns:
        consldt_cols = consolidate_columns(columns=columns)
    return consldt_cols


def get_column_filters(column: exp.Column) -> list[str]:
    """Get the string values that are used to filter a column in a SQL query"""
    # Find if the column is part of a filter expression
    operator = None
    expression = column
    not_str_expr = False
    while expression.parent:
        if operator:
            break
        expression = expression.parent  # type: ignore
        for str_expr in ALL_STRING_EXPRESSION_OPS:
            if isinstance(expression, str_expr):
                operator = str_expr
                found_expr = expression
                break
        for non_str_expr in NON_STRING_EXPRESSION_OPS:
            if isinstance(expression, non_str_expr):
                not_str_expr = True
    if not operator or not_str_expr:
        return []

    # Define function to only return string values
    def check_string(literal: exp.Literal):
        if literal.is_string:
            return literal.this
        return None

    # Collect filter values
    filters = []
    if operator == exp.In:
        for expr in found_expr.expressions:
            filter_val = check_string(expr)
            if filter_val:
                filters.append(filter_val)
    else:
        if found_expr.expression.is_string:
            filters.append(found_expr.expression.this)
    return filters


def get_column_cast(column: exp.Column):
    cast_type = None
    column_cast = column.find_ancestor(exp.Cast)
    if not column_cast:
        date_cast = column.find_ancestor(exp.Date)
        if date_cast:
            return "DATE"
    if column_cast:
        data_type = column_cast.find(exp.DataType)
        if data_type:
            cast_type = data_type.this.value
    return cast_type


def strip_column_qualifiers(expression: Expression) -> Expression:
    """
    Transform a Column expression by removing its table qualifier and alias.
    For other expression types, returns them unchanged.
    """
    if isinstance(expression, exp.Column):
        # Create a new Column with just the column name, no table or alias
        return exp.Column(this=expression.name)

    # For all other expressions, return as is
    return expression


def get_column_func(column: exp.Column):
    not_str_expr = False
    operator = None
    expression = column
    found_expr = None
    while expression.parent:
        if operator:
            break
        expression = expression.parent  # type: ignore
        for str_expr in ALL_STRING_EXPRESSION_OPS:
            if isinstance(expression, str_expr):
                operator = str_expr
                found_expr = expression
                break
        for non_str_expr in NON_STRING_EXPRESSION_OPS:
            if isinstance(expression, non_str_expr):
                not_str_expr = True
    if not operator and not not_str_expr:
        if column.find_ancestor(exp.Where):
            logger.warning(
                """Expression op not found for column despite being in where clause. \
Could be that it's an operator that doesn't deal with strings: %s", column.name"""
            )
        return column.name
    if not found_expr:
        return None
    transformed_col = found_expr.transform(strip_column_qualifiers)
    if not transformed_col:
        return column.name
    return transformed_col.this.sql()


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
    ignore_columns: bool = True,
    ignore_tables: bool = True,
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
                sch.SQLTableColumn(**column.dict()) for column in table.columns if not column.ignore and ignore_columns
            ],
            ignore=table.ignore,
            primary_keys=table.primary_keys,
        )
        for table in tables
        if not table.ignore and ignore_tables
    ]
    return tables_base
