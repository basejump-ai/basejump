import asyncio
import csv
import io
import time
import uuid
from abc import ABC, abstractmethod
from typing import Literal, Optional

import pandas as pd
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result.store import ResultStore, S3ResultStore
from basejump.core.models import models

logger = set_logging(handler_option="stream", name=__name__)


class TableUploader(ABC):
    def __init__(self, result_store: ResultStore, upload_uuid: Optional[uuid.UUID] = None):
        self.result_store = result_store
        if upload_uuid:
            self.result_store.result_uuid = upload_uuid
            self.upload_uuid = upload_uuid
        else:
            self.upload_uuid = self.result_store.result_uuid

    @abstractmethod
    async def upload_file(self, file: UploadFile):
        pass

    @abstractmethod
    async def create_table(
        self,
        db: AsyncSession,
        headers: pd.DataFrame,
        client_id: int,
        db_id: int,
        table_location: str,
        database_name: str,
        table_name: str,
        storage_type: Literal["csv"] = "csv",
    ):
        pass


class S3TableUploader(TableUploader):
    result_store: S3ResultStore

    def __init__(self, result_store: S3ResultStore, upload_uuid: Optional[uuid.UUID] = None):
        super().__init__(result_store=result_store, upload_uuid=upload_uuid)

    @staticmethod
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

    async def upload_file(
        self,
        file: UploadFile,
    ) -> tuple[pd.DataFrame, str]:
        # Upload file
        t1 = time.time()

        # Update the prefix since all files for Athena need to be in a single directory
        self.result_store.prefix = self.get_s3_upload_prefix()

        # Create buffer
        buffer = io.BytesIO()
        text_wrapper = io.TextIOWrapper(buffer, newline="", encoding="utf-8")
        writer = csv.writer(text_wrapper)

        # Stream and write headers
        chunk = await file.read(self.result_store.chunk_size)  # Small chunk to get headers
        text = chunk.decode("utf-8")

        # Initialize multipart upload
        self.result_store.create_multipart_upload()
        headers: list = []

        # Process rest of file
        while chunk:
            if buffer.tell() >= self.result_store.upload_size:
                self.result_store.upload_chunk(buffer=buffer, text_wrapper=text_wrapper)
                buffer.seek(0)
                buffer.truncate()

            for row in csv.reader(text.splitlines()):
                if len(headers) <= 5:
                    headers.append(row)
                writer.writerow(row)

            chunk = await file.read(self.result_store.chunk_size)
            text = chunk.decode("utf-8") if chunk else ""

        # Upload final chunk
        self.result_store.complete_multipart_upload(buffer=buffer, text_wrapper=text_wrapper)
        t2 = time.time()
        logger.debug(f"Time to upload file: {t2-t1}s")
        table_location = self.result_store.get_s3_folder_path(
            bucket_name=self.result_store.bucket_name, prefix=self.result_store.prefix
        )
        return pd.DataFrame(data=headers[1:], columns=headers[0]), table_location

    async def record_table(self, db: AsyncSession, client_id, db_id, table_name, database_name, table_location):
        table_upload = models.TableUpload(
            client_id=client_id,
            upload_uuid=self.upload_uuid,
            table_name=table_name,
            table_location=table_location,
            db_id=db_id,
        )
        db.add(table_upload)
        await db.commit()

    async def create_table(
        self,
        db: AsyncSession,
        headers: pd.DataFrame,
        client_id: int,
        db_id: int,
        table_location: str,
        database_name: str,
        table_name: str,
        storage_type: Literal["csv"] = "csv",
    ):
        t1 = time.time()
        schema = {col: self.get_athena_type(headers[col].dtype) for col in headers.columns.tolist()}
        full_table_name = f"{database_name}.{table_name}"
        # TODO: Doesn't handle headers that are integers. Will get botocore.errorfactory.InvalidRequestException error.
        # Surround cols in backticks.
        create_table_query = f"""\
CREATE EXTERNAL TABLE IF NOT EXISTS {full_table_name} (
{", ".join(f"{str(column)} {dtype}" for column, dtype in schema.items())}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES ('field.delim' = ',')
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' OUTPUTFORMAT \
'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION '{table_location}'
TBLPROPERTIES ('classification' = '{storage_type}','skip.header.line.count'='1');"""
        query_execution = await asyncio.to_thread(
            self.result_store.athena_client.start_query_execution,
            QueryString=create_table_query,
            ResultConfiguration={"OutputLocation": f"s3://{self.result_store.bucket_name}/query_outputs"},
        )
        execution_id = query_execution["QueryExecutionId"]
        query_details = await asyncio.to_thread(
            self.result_store.athena_client.get_query_execution, QueryExecutionId=execution_id
        )

        count = 0
        query_state = query_details["QueryExecution"]["Status"]["State"]
        while query_state not in ["SUCCEEDED", "FAILED", "CANCELED"]:
            # wait n seconds
            time.sleep(1)
            max_time = 10
            if count >= max_time:
                logger.warning("Athena table creation failed after %s seconds", count)
                break
            query_details = await asyncio.to_thread(
                self.result_store.athena_client.get_query_execution, QueryExecutionId=execution_id
            )
            query_state = query_details["QueryExecution"]["Status"]["State"]
            count += 1
        logger_msg = f"Athena table creation {query_state} after {count} seconds"
        logger.info("Athena table location: %s", table_location)
        logger.info("Athena table name: %s", table_name)
        if query_state != "SUCCEEDED":
            logger.warning(logger_msg)
        else:
            logger.info(logger_msg)
            await self.record_table(
                db=db,
                client_id=client_id,
                db_id=db_id,
                table_name=table_name,
                database_name=database_name,
                table_location=table_location,
            )
        t2 = time.time()
        logger.debug(f"Time to create table: {t2-t1}s")

    def get_s3_upload_prefix(self):
        return f"{self.result_store.prefix}{str(self.upload_uuid)}/"
