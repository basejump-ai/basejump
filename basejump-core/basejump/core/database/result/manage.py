"""Manage query results files."""

import asyncio
import io
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import aioboto3
import boto3
import pandas as pd

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import result_utils
from basejump.core.models import constants, errors
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


class ResultManager(ABC):
    def __init__(self, result_file_path: str):
        self.result_file_path = result_file_path

    @abstractmethod
    def get_result(self, max_file_size=constants.MAX_FILE_SIZE) -> pd.DataFrame:
        pass

    @abstractmethod
    async def aget_result(self, max_file_size=constants.MAX_FILE_SIZE) -> pd.DataFrame:
        pass

    @abstractmethod
    def delete_result(self) -> None:
        pass

    @abstractmethod
    async def adelete_result(self) -> None:
        pass

    @abstractmethod
    def stream_result(self):
        pass

    def get_stream_result_generator(self):
        return self.stream_result


class LocalResultManager(ResultManager):
    def __init__(self, result_file_path: str):
        super().__init__(result_file_path=result_file_path)

    def get_result(self, max_file_size: int = constants.MAX_FILE_SIZE) -> pd.DataFrame:
        if max_file_size != constants.MAX_FILE_SIZE:
            logger.warning("Max file size not implemented for LocalResultManager")
        return pd.read_csv(self.result_file_path)

    async def aget_result(self, max_file_size: int = constants.MAX_FILE_SIZE) -> pd.DataFrame:
        if max_file_size != constants.MAX_FILE_SIZE:
            logger.warning("Max file size not implemented for LocalResultManager")
        return pd.read_csv(self.result_file_path)

    def delete_result(self) -> None:
        """Delete the result file from local storage."""
        try:
            os.remove(self.result_file_path)
            logger.info(f"Deleted file: {self.result_file_path}")
        except FileNotFoundError:
            logger.warning(f"File not found: {self.result_file_path}")
            # Decide: raise or ignore (already deleted is success?)
        except PermissionError:
            logger.error(f"Permission denied deleting file: {self.result_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error deleting file {self.result_file_path}: {str(e)}")
            raise

    async def adelete_result(self) -> None:
        """Async wrapper to delete the result file."""
        await asyncio.to_thread(self.delete_result)

    def stream_result(self) -> Iterator:
        try:
            with open(self.result_file_path, "rb") as file:
                yield from file
        except FileNotFoundError:
            logger.error(f"File not found: {self.result_file_path}")
            # You can decide what to do in this case. Here we just return to stop the generator.
            return
        except Exception as e:
            logger.error(f"Error opening file {self.result_file_path}: {str(e)}")
            # You might want to handle other types of exceptions as well.
            return


class S3ResultManager(ResultManager):
    def __init__(self, result_file_path: str, aws_s3_config: sch.AWSS3Config):
        """Manage results from AWS S3.

        Parameters
        ----------
        result_file_path
            Path to the result file in S3.
        aws_s3_config
            AWS S3 configuration object containing connection details.
        max_file_size
            The maximum file size in MB before throwing an error when retrieving results.
        """
        super().__init__(result_file_path=result_file_path)
        self.aws_s3_config = aws_s3_config

    def get_result(self, max_file_size: int = constants.MAX_FILE_SIZE) -> pd.DataFrame:
        raise NotImplementedError("Synchronous get result not implemented for AWS S3.")

    async def aget_result(self, max_file_size: int = constants.MAX_FILE_SIZE) -> pd.DataFrame:
        """Retrieve the result from S3"""
        buffer = io.BytesIO()
        try:
            aws_access_key_id = os.environ["AWS_USER_ACCESS_KEY_ID"]
            aws_secret_access_key = os.environ["AWS_USER_SECRET_ACCESS_KEY"]
            region_name = os.environ["AWS_REGION"]
        except KeyError as e:
            raise errors.MissingEnvironmentVariable(f"Missing an AWS credential environment variable: {str(e)}")
        session = aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        async with session.client("s3") as s3_client:
            key, bucket = self.get_s3_info_from_filepath()
            response = await s3_client.head_object(Bucket=bucket, Key=key)
            file_size = response["ContentLength"]
            if file_size > max_file_size * 1024 * 1024:
                raise errors.FileSizeError
            await s3_client.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            return pd.read_csv(buffer)

    def delete_result(self):
        """Deletes a file from an S3 bucket."""
        s3_client = boto3.client("s3")
        s3_key, bucket_name = self.get_s3_info_from_filepath()
        try:
            # Delete the file from S3
            response = s3_client.delete_object(Bucket=bucket_name, Key=s3_key)

            # Log the result (Optional, for debugging purposes)
            logger.info(f"File {s3_key} deleted successfully from bucket {bucket_name}.")
            return response

        except Exception as e:
            logger.error(f"Error deleting file {s3_key} from bucket {bucket_name}: {e}")
            return None

    async def adelete_result(self):
        s3_key, bucket_name = self.get_s3_info_from_filepath()
        session = aioboto3.Session(
            aws_access_key_id=self.aws_s3_config.access_key,
            aws_secret_access_key=self.aws_s3_config.secret_access_key,
            region_name=self.aws_s3_config.region,
        )
        async with session.client("s3") as s3_client:
            try:
                response = await s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                logger.info(f"File {s3_key} deleted successfully from bucket {bucket_name}.")
                return response
            except Exception as e:
                logger.error(f"Error deleting file {s3_key} from bucket {bucket_name}: {e}")
                return None

    def stream_result(self) -> Iterator:
        """Generator to stream a file from S3."""
        chunk_size = 1024 * 1024
        s3_key, bucket = self.get_s3_info_from_filepath()
        try:
            # Fetch the file from S3
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            file_stream = response["Body"]

            # Read and yield chunks of the file
            while True:
                chunk = file_stream.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        except Exception as e:
            logger.error(f"Error fetching file {s3_key} from S3 bucket {bucket}: {e}")
            # Handle error: You could raise an HTTPException or handle differently
            return

    def get_s3_info_from_filepath(self) -> tuple[str, str]:
        logger.info("Here is the S3 filepath: %s", self.result_file_path)
        try:
            s3_bucket_key = self.result_file_path.split(result_utils.S3_PREFIX)[1]
            file_components = s3_bucket_key.split("/")
            # NOTE: The bucket should be first and the key last, in between is the prefix
            bucket = file_components[0]
            s3_key = "/".join(file_components[1:])
        except Exception:
            raise Exception("Error getting the S3 key and bucket")
        return s3_key, bucket
