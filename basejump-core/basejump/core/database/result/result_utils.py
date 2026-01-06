"""Utilities for query results."""

import pandas as pd
from sqlalchemy.engine import Row

from basejump.core.models import enums
from basejump.core.models import schemas as sch

RESULT_PREVIEW_CT = 100
PREVIEW_SUFFIX = "_preview"
S3_PREFIX = "s3://"


def get_result_type(num_cols: int, num_rows: int) -> enums.ResultType:
    if num_cols == 1 and num_rows in [0, 1]:
        result_type = enums.ResultType.METRIC
    elif num_cols > 1 and num_rows == 1:
        result_type = enums.ResultType.RECORD
    else:
        result_type = enums.ResultType.DATASET

    return result_type


def get_output_df(query_result: list[Row], sql_query: str) -> sch.QueryResultDF:
    # TODO: Have some handling in case this gets too big
    output_df = pd.DataFrame(query_result)
    result_row_ct = len(output_df)
    preview_row_ct = RESULT_PREVIEW_CT if result_row_ct > RESULT_PREVIEW_CT else result_row_ct
    preview_output_df = output_df.head(preview_row_ct)
    num_rows = output_df.shape[0]
    num_cols = output_df.shape[1]
    result_type = get_result_type(num_rows=num_rows, num_cols=num_cols)
    return sch.QueryResultDF(
        output_df=output_df,
        query_result=query_result,
        preview_output_df=preview_output_df,
        preview_row_ct=preview_row_ct,
        num_rows=num_rows,
        num_cols=num_cols,
        result_type=result_type,
        sql_query=sql_query,
    )


def get_preview_file_name(file_path: str) -> str:
    split_file = file_path.split(".csv")
    file_name = split_file[0]
    return f"{file_name}{PREVIEW_SUFFIX}.csv"
