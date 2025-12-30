import asyncio
import copy
import csv
import io
import os
import time
import uuid
from typing import Optional

import boto3
import pandas as pd
import sqlalchemy as sa
from botocore.exceptions import ClientError
from fastapi import UploadFile

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_connection
from basejump.core.database.db_connect import ConnectDB
from basejump.core.database.format_response import JSONResponseFormatter
from basejump.core.models import constants, enums, errors
from basejump.core.models import pydantic_ai_formats as fmt
from basejump.core.models import schemas as sch

RESULT_PREVIEW_CT = 100
PREVIEW_SUFFIX = "_preview"
S3_PREFIX = "s3://"

logger = set_logging(handler_option="stream", name=__name__)


def get_result_type(num_cols: int, num_rows: int) -> enums.ResultType:
    if num_cols == 1 and num_rows in [0, 1]:
        result_type = enums.ResultType.METRIC
    elif num_cols > 1 and num_rows == 1:
        result_type = enums.ResultType.RECORD
    else:
        result_type = enums.ResultType.DATASET

    return result_type


def save_preview(buffer, s3_client, s3_bucket_name, file_name):
    buffer.seek(0)
    logger.info(f"Saving file preview, bucket: {s3_bucket_name}, file_name: {file_name}")
    try:
        s3_client.upload_fileobj(buffer, s3_bucket_name, file_name)
    except ClientError as e:
        logger.error("Error in save_preview %s", str(e))
        raise errors.InvalidClientCredentials


def get_preview_file_name(s3_file_key: str) -> str:
    split_file = s3_file_key.split(".csv")
    file_name = split_file[0]
    return f"{file_name}{PREVIEW_SUFFIX}.csv"


def get_s3_key(file_name, prefix: Optional[str] = None):
    if prefix:
        # NOTE: Prefixes end with a slash
        return f"{prefix}{file_name}"
    return file_name


def get_s3_file_path(s3_file_key: str, bucket_name: str):
    return f"{S3_PREFIX}{bucket_name}/{s3_file_key}"


def get_s3_upload_prefix(prefix: str, result_uuid: uuid.UUID):
    return f"{prefix}{str(result_uuid)}/"


def get_s3_info_from_filepath(filepath) -> tuple[str, str]:
    logger.info("Here is the S3 filepath: %s", filepath)
    try:
        s3_bucket_key = filepath.split(S3_PREFIX)[1]
        file_components = s3_bucket_key.split("/")
        # NOTE: The bucket should be first and the key last, in between is the prefix
        bucket = file_components[0]
        s3_key = "/".join(file_components[1:])
    except Exception:
        raise Exception("Error getting the S3 key and bucket")
    return s3_key, bucket


def get_default_prefix(client_uuid: uuid.UUID):
    return os.environ["AWS_DEFAULT_PREFIX"] + str(client_uuid) + "/"


def get_s3_folder_path(bucket_name: str, prefix: Optional[str] = None):
    if prefix:
        # NOTE: Prefixes end with a slash
        return f"{S3_PREFIX}{bucket_name}/{prefix}"
    return f"{S3_PREFIX}{bucket_name}/"


# TODO: Stream uploads for databases that allow streaming (Redshift does not allow streaming)
class S3Uploader:
    chunk_size = 8192
    upload_size_mb = 5
    upload_size = upload_size_mb * 1024 * 1024
    upload_chunk_limit = 20
    max_upload_size = upload_size * upload_chunk_limit  # 100 MB

    def __init__(
        self,
        client_id: int,
        db_conn_params: sch.SQLDBSchema,
        result_uuid: Optional[uuid.UUID] = None,
        n_rows=5,
        type="sql",
    ):
        self.multipart = False
        self.parts: list[dict] = []
        self.upload_id: Optional[str] = None
        self.top_n_rows: int = n_rows
        self.type: str = type
        self.types = ["csv", "sql"]
        self.result_uuid = result_uuid or uuid.uuid4()
        self.result_file_name = f"{str(self.result_uuid)}.csv"
        assert self.type in self.types, f"'{self.type}' is not in {self.types}"
        self.buffer = io.BytesIO()
        self.text_wrapper = io.TextIOWrapper(self.buffer, newline="", encoding="utf-8")
        self.ai_query_result_view: list = []
        self.saved_preview = False
        self.multipart_upload = False
        self.aborted_upload = False
        self.etags: list = []
        self.counter = 0
        self.chunk_counter = 0
        self.total_row_counter = 0
        self.client_id = client_id
        self.initialize_s3_bucket(db_conn_params=db_conn_params)
        self.metric_value: Optional[str] = None
        self.metric_value_formatted: Optional[str] = None

        logger.info("Uploading result_uuid: %s", str(self.result_uuid))

    @property
    def preview_file_name(self) -> str:
        return get_preview_file_name(self.s3_file_key)

    @property
    def s3_file_key(self) -> str:
        return get_s3_key(file_name=self.result_file_name, prefix=self.prefix)

    def initialize_s3_bucket(self, db_conn_params: sch.SQLDBSchema):
        conn_db = ConnectDB(conn_params=db_conn_params)
        sql_engine_noasync = conn_db.connect_db()
        session = sa.orm.sessionmaker(
            bind=sql_engine_noasync,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        try:
            with session() as connect:
                session_result = crud_connection.get_client_active_storage_conn_sync(
                    db=connect, client_id=self.client_id
                )
                if session_result:
                    self.bucket_name = session_result.bucket_name
                    self.prefix = session_result.prefix
                    self.s3_client = boto3.client(  # type: ignore
                        "s3",
                        region_name=session_result.region,
                        aws_access_key_id=session_result.access_key,
                        aws_secret_access_key=session_result.secret_access_key,
                    )
                    if self.type == "csv":
                        logger.debug("Using region: %s", session_result.region)
                        self.athena_client = boto3.client(  # type: ignore
                            "athena",
                            region_name=session_result.region,
                            aws_access_key_id=session_result.access_key,
                            aws_secret_access_key=session_result.secret_access_key,
                        )
                else:
                    msg = "No bucket name found"
                    logger.info(msg)
                    raise Exception(msg)
        finally:
            sql_engine_noasync.dispose()

    def _upload_chunk(self, part_number):
        self.buffer.seek(0)
        response = self.s3_client.upload_part(
            Bucket=self.bucket_name,
            Key=self.s3_file_key,
            PartNumber=part_number,
            UploadId=self.upload_id,
            Body=self.buffer.getvalue(),
        )
        return response["ETag"]

    def upload_chunk(self):
        self.text_wrapper.flush()
        # If buffer exceeds 5 MB, upload and reset the buffer
        if self.buffer.tell() >= self.chunk_size:
            self.chunk_counter += 1
            if self.chunk_counter > self.upload_chunk_limit:
                # Not allowing uploads past 100 MB currently
                self.abort_multipart_upload()
            else:
                try:
                    etag = self._upload_chunk(part_number=len(self.etags) + 1)
                    self.etags.append(etag)
                    self.buffer.truncate(0)  # Reset the buffer for the next chunk
                except Exception as e:
                    logger.error("Error in upload to s3 in chunks %s", str(e))
                    # Not raising error since this could also indicate it completed

    def clean_row(self, row):
        return [str(cell).replace("\n", "\\n").replace("\r", "") for cell in row]

    def create_multipart_upload(self):
        try:
            multipart_upload = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name, Key=self.s3_file_key, ContentType="text/csv"
            )
        except Exception as e:
            logger.error("Error in stream query results %s", str(e))
            raise e
        self.upload_id = multipart_upload["UploadId"]

    def complete_multipart_upload(self):
        self.text_wrapper.flush()
        # Upload the last part if there is any data remaining
        if self.buffer.tell() > 0:
            etag = self._upload_chunk(
                part_number=len(self.etags) + 1,
            )
            self.etags.append(etag)
        # Complete the multipart upload
        self.s3_client.complete_multipart_upload(
            Bucket=self.bucket_name,
            Key=self.s3_file_key,
            UploadId=self.upload_id,
            MultipartUpload={"Parts": [{"PartNumber": idx + 1, "ETag": etag} for idx, etag in enumerate(self.etags)]},
        )

    def abort_multipart_upload(self):
        self.s3_client.abort_multipart_upload(Bucket=self.bucket_name, Key=self.s3_file_key, UploadId=self.upload_id)
        # Confirm all parts are deleted
        try:
            parts = self.s3_client.list_parts(Bucket=self.bucket_name, Key=self.s3_file_key, UploadId=self.upload_id)
        except ClientError as e:
            logger.warning("Error when listing parts %s", str(e))
            raise e
        # If parts still exist, then try to abort again
        if len(parts["Parts"]) > 0:
            try:
                self.s3_client.abort_multipart_upload(
                    Bucket=self.bucket_name, Key=self.s3_file_key, UploadId=self.upload_id
                )
            except Exception as e:
                logger.warning("Error when in multipart upload %s", str(e))
                raise e

    def single_upload(self):
        self.text_wrapper.flush()
        if not self.saved_preview:
            preview_buffer_to_upload = copy.deepcopy(self.buffer)
            save_preview(preview_buffer_to_upload, self.s3_client, self.bucket_name, self.preview_file_name)
            self.saved_preview = True
        self.buffer.seek(0)
        buffer_to_upload = copy.deepcopy(self.buffer)
        try:
            self.s3_client.upload_fileobj(buffer_to_upload, self.bucket_name, self.s3_file_key)
        except ClientError as e:
            logger.error("Invalid client creds: %s", str(e))
            raise errors.InvalidClientCredentials
        assert self.saved_preview

    def save_preview(self):
        self.text_wrapper.flush()
        buffer_to_upload = copy.deepcopy(self.buffer)
        save_preview(
            buffer=buffer_to_upload,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket_name,
            file_name=get_preview_file_name(self.s3_file_key),
        )
        self.saved_preview = True

    def get_metric_value(self, small_model_info: sch.ModelInfo, initial_prompt: str, sql_query: str):
        # Get the metric values
        # TODO: Possibly stop saving metrics in S3 since we're saving them in ResultHistory now
        # Doing this does cause issues elsewhere in the code though, so needs to be done carefully
        metric_value_binary = self.buffer.getvalue()
        self.metric_value = str(metric_value_binary.decode().replace("\n", " ").replace("\r", "").strip())
        prompt = f"""\
Update the following metric value to be formatted based on the context. \
You are given the original user prompt, the SQL query to answer the prompt, \
and the metric value. A few examples to help explain:
- If dealing with currency, add the correct currency symbol. An example is formatting a \
metric value of 4000 to $4,000 (assume US currency if speaking English).
- Adding commas for values over 1,000.
- Adding a unit of measurement if appropriate. For example, if the metric is describing \
the number of bricks and the value was 100, then 100 would be reformatted to 100 bricks instead.
Prompt: {initial_prompt}\n
SQL Query: {sql_query}\n
Metric Value: {self.metric_value}\n
"""
        format_json_response = JSONResponseFormatter(
            small_model_info=small_model_info, response=prompt, pydantic_format=fmt.FormattedMetric
        )
        extract = format_json_response.format_sync()
        self.metric_value = str(extract.metric_value)
        self.metric_value_formatted = extract.metric_value_formatted

    def upload_sql_result(
        self, result: sa.engine.CursorResult, small_model_info: sch.ModelInfo, initial_prompt: str, sql_query: str
    ) -> sch.QueryResult:
        # Create a CSV writer that writes into the buffer
        csv_writer = csv.writer(self.text_wrapper)

        # Write the header
        self.cols = result.keys()
        csv_writer.writerow(self.cols)  # Write column names as header

        # Process rows one by one and upload in chunks
        # HACK: Use pagination since server-side cursors aren't available for redshift
        for row in result:
            if self.counter <= constants.AI_RESULT_PREVIEW_CT:
                self.ai_query_result_view.append(row)
            self.counter += 1
            self.total_row_counter += 1
            cleaned_row = self.clean_row(row)  # Clean the row to handle newlines
            csv_writer.writerow(cleaned_row)

            # Save the preview if it hasn't been saved
            if self.counter == 100 and not self.saved_preview:
                self.save_preview()

            # Flush the underlying buffer after writing - only flush every 500 rows to improve performance
            if self.counter > self.chunk_size:
                self.counter = 0
                if not self.multipart_upload:
                    self.create_multipart_upload()
                self.upload_chunk()

        # Complete the multipart upload
        if self.multipart_upload:
            self.complete_multipart_upload()
        else:
            # Otherwise use a single upload
            self.single_upload()
        if self.counter == 1:
            self.get_metric_value(
                small_model_info=small_model_info, initial_prompt=initial_prompt, sql_query=sql_query
            )
        return self.create_query_result(sql_query=sql_query)

    def create_query_result(self, sql_query: str) -> sch.QueryResult:
        preview_row_ct = RESULT_PREVIEW_CT if self.counter > RESULT_PREVIEW_CT else self.counter
        num_rows = self.total_row_counter
        num_cols = len(self.cols)
        result_type = get_result_type(num_rows=num_rows, num_cols=num_cols)
        file_size_est_base = self.upload_size * self.chunk_counter
        file_size_est = f"<{self.upload_size_mb}MB" if file_size_est_base == 0 else f"{file_size_est_base}MB"
        logger.debug(f"File has {num_rows} rows and {num_cols} columns. Estimated file size is {file_size_est}")
        result_file_path = get_s3_file_path(s3_file_key=self.s3_file_key, bucket_name=self.bucket_name)
        preview_file_path = get_s3_file_path(s3_file_key=self.preview_file_name, bucket_name=self.bucket_name)
        logger.info("Here is the result file path: %s", result_file_path)
        logger.info("Here is the result preview file path: %s", preview_file_path)
        return sch.QueryResult(
            result_uuid=self.result_uuid,
            preview_row_ct=preview_row_ct,
            query_result=self.ai_query_result_view[: constants.AI_RESULT_PREVIEW_CT],  # Adding as extra safeguard
            ai_preview_row_ct=constants.AI_RESULT_PREVIEW_CT,
            num_rows=num_rows,
            num_cols=len(self.cols),
            result_file_path=result_file_path,
            preview_file_path=preview_file_path,
            result_type=result_type,
            sql_query=sql_query,
            metric_value=self.metric_value,
            metric_value_formatted=self.metric_value_formatted,
            aborted_upload=self.aborted_upload,
        )

    async def upload_file(self, file: UploadFile) -> pd.DataFrame:
        # Update the prefix since all files for Athena need to be in a single directory
        self.prefix = get_s3_upload_prefix(prefix=self.prefix, result_uuid=self.result_uuid)

        # Create buffer
        buffer = io.StringIO()
        writer = csv.writer(self.text_wrapper)

        # Stream and write headers
        chunk = await file.read(self.chunk_size)  # Small chunk to get headers
        text = chunk.decode("utf-8")

        # Initialize multipart upload
        self.create_multipart_upload()
        headers: list = []

        # Process rest of file
        while chunk:
            if buffer.tell() >= self.upload_size:
                self.upload_chunk()
                writer = csv.writer(buffer)

            for row in csv.reader(text.splitlines()):
                if len(headers) <= 5:
                    headers.append(row)
                writer.writerow(row)

            chunk = await file.read(self.chunk_size)
            text = chunk.decode("utf-8") if chunk else ""

        # Upload final chunk
        self.complete_multipart_upload()
        return pd.DataFrame(data=headers[1:], columns=headers[0])

    def _create_table_from_csv(self, headers: pd.DataFrame):
        schema = {col: get_athena_type(headers[col].dtype) for col in headers.columns.tolist()}
        table_suffix = str(copy.copy(self.result_uuid)).replace("-", "_")
        table_location = get_s3_folder_path(prefix=self.prefix, bucket_name=self.bucket_name)
        table_name = f"default.uploaded_table_{table_suffix}"
        # TODO: Doesn't handle headers that are integers. Will get botocore.errorfactory.InvalidRequestException error.
        # Surround cols in backticks.
        create_table = f"""\
CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (
{", ".join(f"{str(column)} {dtype}" for column, dtype in schema.items())}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES ('field.delim' = ',')
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' OUTPUTFORMAT \
'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION '{table_location}'
TBLPROPERTIES ('classification' = 'csv','skip.header.line.count'='1');"""
        query_execution = self.athena_client.start_query_execution(
            QueryString=create_table, ResultConfiguration={"OutputLocation": f"s3://{self.bucket_name}/query_outputs"}
        )
        execution_id = query_execution["QueryExecutionId"]
        query_details = self.athena_client.get_query_execution(QueryExecutionId=execution_id)

        count = 0
        query_state = query_details["QueryExecution"]["Status"]["State"]
        while query_state not in ["SUCCEEDED", "FAILED", "CANCELED"]:
            # wait n seconds
            time.sleep(1)
            max_time = 30
            if count >= max_time:
                logger.warning("Athena table creation failed after %s seconds", count)
                break
            query_details = self.athena_client.get_query_execution(QueryExecutionId=execution_id)
            query_state = query_details["QueryExecution"]["Status"]["State"]
            count += 1
        logger_msg = f"Athena table creation {query_state} after {count} seconds"
        logger.info("Athena table location: %s", table_location)
        logger.info("Athena table name: %s", table_name)
        if query_state != "SUCCEEDED":
            logger.warning(logger_msg)
        else:
            logger.info(logger_msg)


def get_athena_type(pd_type) -> str:
    type_map = {
        # Numeric
        "int8": "tinyint",
        "int16": "smallint",
        "int32": "int",
        "int64": "bigint",
        "uint8": "smallint",
        "uint16": "int",
        "uint32": "bigint",
        "uint64": "decimal",
        "float16": "float",
        "float32": "float",
        "float64": "double",
        "decimal": "decimal",
        # String/Text
        "object": "string",
        "string": "string",
        "category": "string",
        # Boolean
        "bool": "boolean",
        # Date/Time
        "datetime64[ns]": "timestamp",
        "datetime64[ms]": "timestamp",
        "datetime64[us]": "timestamp",
        "timedelta64[ns]": "string",
        "date": "date",
        "time": "string",
    }
    return type_map.get(pd_type, "string")


async def upload_csv_to_s3(
    db_conn_params: sch.SQLDBSchema, file: UploadFile, client_id: int, result_uuid: Optional[uuid.UUID] = None
) -> sch.UploadResult:
    result_uuid = result_uuid or uuid.uuid4()
    uploader = S3Uploader(
        db_conn_params=db_conn_params,
        client_id=client_id,
        type="csv",
        result_uuid=result_uuid,
    )

    # Upload file
    t1 = time.time()
    headers = await uploader.upload_file(file=file)
    t2 = time.time()
    logger.debug(f"Time to upload file: {t2-t1}s")
    # Get datatypes and create table
    t1 = time.time()
    await asyncio.to_thread(uploader._create_table_from_csv, headers=headers)
    t2 = time.time()
    logger.debug(f"Time to create table: {t2-t1}s")
    return sch.UploadResult(result_uuid=result_uuid, s3_file_key=uploader.s3_file_key)


def upload_sql_to_s3(
    conn: sa.Connection,
    db_conn_params: sch.SQLDBSchema,
    sql_query: str,
    initial_prompt: str,
    client_id: int,
    small_model_info: sch.ModelInfo,
    result_uuid: Optional[uuid.UUID] = None,
) -> sch.QueryResult:
    uploader = S3Uploader(
        db_conn_params=db_conn_params,
        client_id=client_id,
        type="sql",
        result_uuid=result_uuid,
    )
    with conn.execute(sa.text(sql_query)) as result:
        upload_result = uploader.upload_sql_result(
            result=result, small_model_info=small_model_info, initial_prompt=initial_prompt, sql_query=sql_query
        )
    return upload_result
