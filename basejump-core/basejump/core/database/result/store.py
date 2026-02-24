"""Result storage for query results."""

import copy
import csv
import io
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

import boto3
import sqlalchemy as sa
from botocore.exceptions import ClientError

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import manager
from basejump.core.database.result import utils as result_utils
from basejump.core.models import constants, errors
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai import formatter

logger = set_logging(handler_option="stream", name=__name__)

PREVIEW_SUFFIX = "_preview"


class ResultStore(ABC):
    def __init__(
        self,
        client_id: int,
        n_rows=5,
    ):
        self.top_n_rows: int = n_rows
        self.ai_query_result_view: list = []
        self.saved_preview = False
        self.counter = 0
        self.chunk_counter = 0
        self.total_row_counter = 0
        self.client_id = client_id
        self.aborted_upload = False
        self.metric_value: Optional[str] = None
        self.metric_value_formatted: Optional[str] = None

    @abstractmethod
    def get_file_name(self, uuid: UUID) -> str:
        return self.add_file_extension(file_name=str(uuid))

    @abstractmethod
    def get_file_path(self, file_name: str) -> str:
        pass

    @abstractmethod
    def get_preview_file_name(self, uuid: UUID) -> str:
        file_name = self.get_file_name(uuid=uuid)
        split_file = file_name.split(".csv")
        file_name = split_file[0]
        return f"{file_name}{PREVIEW_SUFFIX}.csv"

    @abstractmethod
    def get_preview_file_path(self, preview_file_name: str) -> str:
        return self.get_file_path(file_name=preview_file_name)

    @abstractmethod
    def get_file_info(self, result_uuid: UUID) -> sch.ResultFileInfo:
        pass

    @abstractmethod
    def store(
        self,
        file_info: sch.ResultFileInfo,
        result: sa.engine.CursorResult,
        small_model_info: sch.ModelInfo,
        initial_prompt: str,
        sql_query: str,
    ):
        pass

    @abstractmethod
    def get_result_manager(self, result_file_path: str):
        pass

    @abstractmethod
    def save_preview(
        self,
        preview_file_name: str,
        buffer: io.BytesIO,
        text_wrapper: io.TextIOWrapper,
        preview_file_path: Optional[str] = None,
    ):
        pass

    @staticmethod
    def add_file_extension(file_name: str) -> str:
        """Adds the file extension to the file name"""
        file_name = f"{file_name}.csv"
        return file_name

    def reset(self):
        self.ai_query_result_view = []
        self.saved_preview = False
        self.counter = 0
        self.chunk_counter = 0
        self.total_row_counter = 0
        self.aborted_upload = False
        self.metric_value = None
        self.metric_value_formatted = None

    def create_file_info(self, file_name: str, file_path: str, result_uuid: UUID) -> sch.ResultFileInfo:
        preview_file_name = self.get_preview_file_name(uuid=result_uuid)
        preview_file_path = self.get_preview_file_path(preview_file_name=preview_file_name)
        return sch.ResultFileInfo(
            file_name=file_name,
            file_path=file_path,
            preview_file_name=preview_file_name,
            preview_file_path=preview_file_path,
            result_uuid=result_uuid,
        )

    def create_query_result(
        self, sql_query: str, file_info: sch.ResultFileInfo, columns: sa.engine.result.RMKeyView
    ) -> sch.QueryResult:
        preview_row_ct = (
            result_utils.RESULT_PREVIEW_CT if self.counter > result_utils.RESULT_PREVIEW_CT else self.counter
        )
        num_rows = self.total_row_counter
        num_cols = len(columns)
        logger.debug(f"File has {num_rows} rows and {num_cols} columns.")
        result_type = result_utils.get_result_type(num_rows=num_rows, num_cols=num_cols)
        logger.info("Here is the result file path: %s", file_info.file_path)
        logger.info("Here is the result preview file path: %s", file_info.preview_file_path)
        return sch.QueryResult(
            result_uuid=file_info.result_uuid,
            preview_row_ct=preview_row_ct,
            query_result=self.ai_query_result_view[: constants.AI_RESULT_PREVIEW_CT],  # Adding as extra safeguard
            ai_preview_row_ct=constants.AI_RESULT_PREVIEW_CT,
            num_rows=num_rows,
            num_cols=len(columns),
            result_file_path=file_info.file_path,
            preview_file_path=file_info.preview_file_path,
            result_type=result_type,
            sql_query=sql_query,
            metric_value=self.metric_value,
            metric_value_formatted=self.metric_value_formatted,
            aborted_upload=self.aborted_upload,
        )

    def get_metric_value(
        self, small_model_info: sch.ModelInfo, initial_prompt: str, sql_query: str, buffer: io.BytesIO
    ):
        # Get the metric values
        # TODO: Possibly stop saving metrics in S3 since we're saving them in ResultHistory now
        # Doing this does cause issues elsewhere in the code though, so needs to be done carefully
        metric_value_binary = buffer.getvalue()
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
        format_json_response = formatter.JSONResponseFormatter(
            small_model_info=small_model_info, response=prompt, pydantic_format=fmt.FormattedMetric
        )
        extract = format_json_response.format_sync()
        self.metric_value = str(extract.metric_value)
        self.metric_value_formatted = extract.metric_value_formatted

    def clean_row(self, row):
        return [str(cell).replace("\n", "\\n").replace("\r", "") for cell in row]


class LocalResultStore(ResultStore):
    def __init__(
        self,
        client_id: int,
        n_rows=5,
    ):
        super().__init__(
            client_id=client_id,
            n_rows=n_rows,
        )

    def get_file_name(self, uuid: UUID) -> str:
        return super().get_file_name(uuid=uuid)

    def get_file_path(self, file_name: str) -> str:
        # Get the output directory
        output_dir = pathlib.Path("./data/sql_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / file_name
        return file_path.as_posix()

    def get_preview_file_name(self, uuid: UUID) -> str:
        return super().get_preview_file_name(uuid=uuid)

    def get_preview_file_path(self, preview_file_name: str) -> str:
        return self.get_file_path(file_name=preview_file_name)

    def get_file_info(self, result_uuid: UUID) -> sch.ResultFileInfo:
        file_name = self.get_file_name(uuid=result_uuid)
        file_path = self.get_file_path(file_name=file_name)
        return super().create_file_info(file_name=file_name, file_path=file_path, result_uuid=result_uuid)

    def store(
        self,
        file_info: sch.ResultFileInfo,
        result: sa.engine.CursorResult,
        small_model_info: sch.ModelInfo,
        initial_prompt: str,
        sql_query: str,
    ) -> sch.QueryResult:
        # HACK: Really just need a new result store per query instead
        super().reset()
        logger.info("Saving result_uuid to local storage: %s", str(file_info.result_uuid))

        buffer = io.BytesIO()
        text_wrapper = io.TextIOWrapper(buffer, newline="", encoding="utf-8")
        # Open local CSV file
        # Create a CSV writer that writes into the buffer
        csv_writer = csv.writer(text_wrapper)

        # Write the header
        columns = result.keys()
        csv_writer.writerow(columns)

        # Reset counters for local write
        self.counter = 0
        self.total_row_counter = 0
        self.ai_query_result_view = []
        # Process rows and write directly to file
        for row in result:
            if self.counter <= constants.AI_RESULT_PREVIEW_CT:
                self.ai_query_result_view.append(row)
            self.counter += 1
            self.total_row_counter += 1

            cleaned_row = self.clean_row(row)
            csv_writer.writerow(cleaned_row)

            # Save the preview if it hasn't been saved
            if self.counter == 100 and not self.saved_preview:
                self.save_preview(
                    preview_file_name=file_info.preview_file_name, buffer=buffer, text_wrapper=text_wrapper
                )

        # If exactly one row, compute metric as before
        if self.counter == 1:
            self.get_metric_value(
                small_model_info=small_model_info, initial_prompt=initial_prompt, sql_query=sql_query, buffer=buffer
            )

        # Save the preview
        if not self.saved_preview:
            self.save_preview(preview_file_name=file_info.preview_file_name, buffer=buffer, text_wrapper=text_wrapper)

        # Save the result
        text_wrapper.flush()
        buffer.seek(0)
        data = buffer.getvalue()
        with open(file_info.file_path, "wb") as f:
            f.write(data)

        # Get the query result
        return self.create_query_result(
            sql_query=sql_query,
            file_info=file_info,
            columns=columns,
        )

    def get_result_manager(self, result_file_path: str):
        return manager.LocalResultManager(result_file_path=result_file_path)

    def save_preview(
        self,
        preview_file_name: str,
        buffer: io.BytesIO,
        text_wrapper: io.TextIOWrapper,
        preview_file_path: Optional[str] = None,
    ):
        if not preview_file_path:
            preview_file_path = self.get_preview_file_path(preview_file_name=preview_file_name)
        text_wrapper.flush()
        buffer_to_upload = copy.deepcopy(buffer)
        buffer_to_upload.seek(0)
        data = buffer_to_upload.getvalue()
        with open(preview_file_path, "wb") as f:
            f.write(data)
        self.saved_preview = True


# TODO: Stream uploads for databases that allow streaming (Redshift does not allow streaming)
class S3ResultStore(ResultStore):
    chunk_size = 1024 * 100
    bytes_in_a_mb = 1024 * 1024
    upload_size = 5 * bytes_in_a_mb  # S3 requires a minimum of a 5MB upload size per part
    upload_limit_in_mb = 0.05
    upload_chunk_limit = upload_limit_in_mb * bytes_in_a_mb / chunk_size

    def __init__(
        self,
        client_id: int,
        aws_s3_config: sch.AWSS3Config,
        n_rows=5,
    ):
        super().__init__(
            client_id=client_id,
            n_rows=n_rows,
        )
        self.multipart = False
        self.parts: list[dict] = []
        self.upload_id: Optional[str] = None
        self.multipart_upload = False
        self.etags: list = []
        self.aws_s3_config = aws_s3_config
        self.prefix = self.aws_s3_config.prefix
        self.bucket_name = self.aws_s3_config.bucket_name
        self.region = self.aws_s3_config.region
        self.access_key_id = self.aws_s3_config.access_key
        self.secret_access_key = self.aws_s3_config.secret_access_key
        self.s3_client = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )
        self.athena_client = boto3.client(
            "athena",
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

    @property
    def current_file_size(self):
        return self.chunk_counter * self.chunk_size / (1024 * 1024)

    @staticmethod
    def get_default_prefix(client_uuid: UUID) -> str:
        default_prefix = os.getenv("AWS_DEFAULT_PREFIX") or ""
        return default_prefix + str(client_uuid) + "/"

    @classmethod
    def get_folder_path(cls, bucket_name: str, prefix: str):
        if prefix:
            # NOTE: Prefixes end with a slash
            return f"{result_utils.S3_PREFIX}{bucket_name}/{prefix}"
        return f"{result_utils.S3_PREFIX}{bucket_name}/"

    def get_file_path(self, file_name: str) -> str:
        return result_utils.get_s3_file_path(s3_file_key=file_name, bucket_name=self.bucket_name)

    def get_file_name(self, uuid: UUID) -> str:
        file_name = super().get_file_name(uuid=uuid)
        if self.prefix:
            # NOTE: Prefixes end with a slash
            return f"{self.prefix}{file_name}"
        return file_name

    def get_preview_file_name(self, uuid: UUID) -> str:
        return super().get_preview_file_name(uuid=uuid)

    def get_preview_file_path(self, preview_file_name: str) -> str:
        return self.get_file_path(file_name=preview_file_name)

    def get_file_info(self, result_uuid: UUID) -> sch.ResultFileInfo:
        file_name = self.get_file_name(uuid=result_uuid)
        file_path = self.get_file_path(file_name=file_name)
        return super().create_file_info(file_name=file_name, file_path=file_path, result_uuid=result_uuid)

    def store(
        self,
        file_info: sch.ResultFileInfo,
        result: sa.engine.CursorResult,
        small_model_info: sch.ModelInfo,
        initial_prompt: str,
        sql_query: str,
    ) -> sch.QueryResult:
        super().reset()
        logger.info("Uploading result_uuid to S3: %s", str(file_info.result_uuid))

        buffer = io.BytesIO()
        text_wrapper = io.TextIOWrapper(buffer, newline="", encoding="utf-8")
        # Create a CSV writer that writes into the buffer
        csv_writer = csv.writer(text_wrapper)

        # Write the header
        columns = result.keys()
        csv_writer.writerow(columns)  # Write column names as header

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
                self.save_preview(
                    preview_file_name=file_info.preview_file_name,
                    preview_file_path=file_info.preview_file_path,
                    buffer=buffer,
                    text_wrapper=text_wrapper,
                )

            # Flush the underlying buffer after writing
            if self.counter > self.chunk_size:
                logger.debug(f"Chunk counter at {self.chunk_counter}, reached chunk size of {self.chunk_size}")
                self.counter = 0
                self.upload_chunk(s3_file_key=file_info.file_name, buffer=buffer, text_wrapper=text_wrapper)

        # Complete the multipart upload
        if self.multipart_upload:
            self.complete_multipart_upload(s3_file_key=file_info.file_name, buffer=buffer, text_wrapper=text_wrapper)
        else:
            # Otherwise use a single upload
            self.single_upload(
                s3_file_key=file_info.file_name,
                buffer=buffer,
                text_wrapper=text_wrapper,
                preview_file_name=file_info.preview_file_name,
            )
        if self.counter == 1:
            self.get_metric_value(
                small_model_info=small_model_info, initial_prompt=initial_prompt, sql_query=sql_query, buffer=buffer
            )
        if self.current_file_size == 0:
            file_size_est = f"<{round(self.chunk_size / (1024 * 1024), 2)}"
        else:
            file_size_est = round(self.current_file_size, 2)
        logger.debug(f"Estimated file size is {file_size_est}MB")
        return self.create_query_result(
            sql_query=sql_query,
            file_info=file_info,
            columns=columns,
        )

    def get_result_manager(self, result_file_path: str):
        return manager.S3ResultManager(result_file_path=result_file_path, aws_s3_config=self.aws_s3_config)

    def _upload_chunk(self, s3_file_key: str, part_number, buffer: io.BytesIO):
        buffer.seek(0)
        response = self.s3_client.upload_part(
            Bucket=self.bucket_name,
            Key=s3_file_key,
            PartNumber=part_number,
            UploadId=self.upload_id,
            Body=buffer.getvalue(),
        )
        return response["ETag"]

    def upload_chunk(self, s3_file_key: str, buffer: io.BytesIO, text_wrapper: io.TextIOWrapper):
        if text_wrapper:
            text_wrapper.flush()
        # Upload and reset the buffer
        if buffer.tell() >= self.upload_size:
            if not self.multipart_upload:
                self.create_multipart_upload(s3_file_key=s3_file_key)
            self.chunk_counter += 1
            logger.debug(f"Chunk counter at {self.chunk_counter} and file size of {self.current_file_size}MB.")
            if self.current_file_size > self.upload_chunk_limit:
                # Not allowing uploads past 10 MB currently
                self.abort_multipart_upload(s3_file_key=s3_file_key)
            else:
                try:
                    etag = self._upload_chunk(s3_file_key=s3_file_key, part_number=len(self.etags) + 1, buffer=buffer)
                    self.etags.append(etag)
                    buffer.truncate(0)  # Reset the buffer for the next chunk
                except Exception as e:
                    logger.error("Error in upload to s3 in chunks %s", str(e))
                    # Not raising error since this could also indicate it completed

    def create_multipart_upload(self, s3_file_key: str):
        try:
            multipart_upload = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name, Key=s3_file_key, ContentType="text/csv"
            )
        except Exception as e:
            logger.error("Error in stream query results %s", str(e))
            raise e
        self.upload_id = multipart_upload["UploadId"]
        self.multipart_upload = True

    def complete_multipart_upload(self, s3_file_key: str, buffer: io.BytesIO, text_wrapper: io.TextIOWrapper):
        text_wrapper.flush()
        # Upload the last part if there is any data remaining
        if buffer.tell() > 0:
            etag = self._upload_chunk(s3_file_key=s3_file_key, part_number=len(self.etags) + 1, buffer=buffer)
            self.etags.append(etag)
        # Complete the multipart upload
        self.s3_client.complete_multipart_upload(
            Bucket=self.bucket_name,
            Key=s3_file_key,
            UploadId=self.upload_id,
            MultipartUpload={"Parts": [{"PartNumber": idx + 1, "ETag": etag} for idx, etag in enumerate(self.etags)]},
        )

    def abort_multipart_upload(self, s3_file_key: str) -> None:
        self.aborted_upload = True
        self.s3_client.abort_multipart_upload(Bucket=self.bucket_name, Key=s3_file_key, UploadId=self.upload_id)
        # Confirm all parts are deleted
        try:
            try:
                parts = self.s3_client.list_parts(Bucket=self.bucket_name, Key=s3_file_key, UploadId=self.upload_id)
            except ClientError as e:
                logger.warning("Error when listing parts %s", str(e))
                raise e
            # If parts still exist, then try to abort again
            if len(parts["Parts"]) > 0:
                try:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket_name, Key=s3_file_key, UploadId=self.upload_id
                    )
                except Exception as e:
                    logger.warning("Error when in multipart upload %s", str(e))
                    raise e
        except Exception as e:
            logger.warning(f"Error when aborting multipart upload: {str(e)}")
        finally:
            raise errors.AbortMultipartUpload(max_file_size=f"{self.upload_limit_in_mb} MB")

    def _save_preview(self, buffer, s3_client, s3_bucket_name, file_name):
        buffer.seek(0)
        logger.info(f"Saving file preview, bucket: {s3_bucket_name}, file_name: {file_name}")
        try:
            s3_client.upload_fileobj(buffer, s3_bucket_name, file_name)
        except ClientError as e:
            logger.error("Error in save_preview %s", str(e))
            raise errors.InvalidClientCredentials

    def single_upload(
        self, s3_file_key: str, preview_file_name: str, buffer: io.BytesIO, text_wrapper: io.TextIOWrapper
    ):
        text_wrapper.flush()
        if not self.saved_preview:
            preview_buffer_to_upload = copy.deepcopy(buffer)
            self._save_preview(
                buffer=preview_buffer_to_upload,
                s3_client=self.s3_client,
                s3_bucket_name=self.bucket_name,
                file_name=preview_file_name,
            )
            self.saved_preview = True
        buffer.seek(0)
        buffer_to_upload = copy.deepcopy(buffer)
        try:
            self.s3_client.upload_fileobj(buffer_to_upload, self.bucket_name, s3_file_key)
        except ClientError as e:
            logger.error("Invalid client creds: %s", str(e))
            raise errors.InvalidClientCredentials
        assert self.saved_preview

    def save_preview(
        self,
        preview_file_name: str,
        buffer: io.BytesIO,
        text_wrapper: io.TextIOWrapper,
        preview_file_path: Optional[str] = None,
    ):
        text_wrapper.flush()
        buffer_to_upload = copy.deepcopy(buffer)
        self._save_preview(
            buffer=buffer_to_upload,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket_name,
            file_name=preview_file_name,
        )
        self.saved_preview = True
