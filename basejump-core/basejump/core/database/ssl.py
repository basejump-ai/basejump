import os
import ssl
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy import Engine

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums, errors

logger = set_logging(handler_option="stream", name=__name__)


class SSLParams(ABC):
    def __init__(self, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        self.ssl_mode = ssl_mode
        self.ssl_root_cert = ssl_root_cert

    def get_cert(self):
        if not self.ssl_root_cert:
            raise errors.SSLConfigError
        # Create a temporary file for the root cert
        named_file = tempfile.NamedTemporaryFile(suffix=".crt", delete=False)
        named_file.write(bytes(self.ssl_root_cert.encode()))
        named_file.close()
        return named_file.name

    def get_ssl_args(self) -> tuple:
        if self.ssl_mode == enums.SSLModes.REQUIRE:
            return self.get_require()
        elif self.ssl_mode == enums.SSLModes.VERIFY_CA:
            return self.get_verify_ca()
        elif self.ssl_mode == enums.SSLModes.VERIFY_FULL:
            return self.get_verify_full()
        else:
            raise NotImplementedError(f"SSL mode '{self.ssl_mode.value}' is not supported")

    @abstractmethod
    def get_require(self) -> tuple:
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        pass

    @abstractmethod
    def get_verify_ca(self) -> tuple:
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        pass

    @abstractmethod
    def get_verify_full(self) -> tuple:
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        pass


class BaseSSLContextParams(SSLParams):
    def __init__(self, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        super().__init__(ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)

    def get_require(self) -> tuple:
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return {"ssl": ssl_context}, None

    def get_verify_ca(self) -> tuple:
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        ssl_cert_path = self.get_cert()
        ssl_context = ssl.create_default_context(cafile=ssl_cert_path)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return {"ssl": ssl_context}, ssl_cert_path

    def get_verify_full(self) -> tuple:
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        ssl_cert_path = self.get_cert()
        ssl_context = ssl.create_default_context(cafile=ssl_cert_path)
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return {"ssl": ssl_context}, ssl_cert_path


class MySQLSSLParams(BaseSSLContextParams):
    pass


class SnowflakeSSLParams(BaseSSLContextParams):
    pass


class AthenaSSLParams(BaseSSLContextParams):
    pass


class PostgresSSLParams(SSLParams):
    def __init__(self, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        super().__init__(ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)

    def get_require(self):
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        return {"sslmode": self.ssl_mode.value}, None

    def get_verify_ca(self):
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        ssl_cert_path = self.get_cert()
        return {"sslmode": self.ssl_mode.value, "sslrootcert": ssl_cert_path}, ssl_cert_path

    def get_verify_full(self):
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        ssl_cert_path = self.get_cert()
        return {"sslmode": self.ssl_mode.value, "sslrootcert": ssl_cert_path}, ssl_cert_path


class MSSQLSSLParams(SSLParams):
    def __init__(self, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        super().__init__(ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)

    def get_require(self) -> tuple:
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        return {"encrypt": "YES", "trust_server_certificate": "YES"}, None

    def get_verify_ca(self) -> tuple:
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        # NOTE: Use verify full since no distinction to not check hostname
        return self.get_verify_full()

    def get_verify_full(self) -> tuple:
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        # NOTE: MS Certs are already trusted and installed so no path is necessary
        return {"encrypt": "YES", "trust_server_certificate": "No"}, None


class SSLEngine(Engine):
    """Light wrapper to handle SSL certificate cleanup"""

    def __init__(self, original_engine: Engine, ssl_cert_path: Optional[str]):
        # Copy all attributes from the original engine
        self.__dict__.update(original_engine.__dict__)
        self._original_engine = original_engine
        self.ssl_cert_path = ssl_cert_path

    def __del__(self):
        """Ensure cleanup of temp file if context manager wasn't used"""
        if self.ssl_cert_path:
            try:
                os.remove(self.ssl_cert_path)
            except FileNotFoundError as e:
                logger.warning("File not found %s", str(e))
                pass
