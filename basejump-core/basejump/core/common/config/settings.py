from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

settings_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    env_prefix="BASEJUMP_",
)


class Settings(BaseSettings):
    # Relational database
    db_user: str = "localuser"
    db_password: SecretStr = SecretStr("localuser123")
    db_host: str = "localhost"
    db_name: str = "localdb"
    db_port: int = 5432
    encryption_key: Optional[SecretStr] = None
    ssl: bool = False  # should always be True in production
    # Vector database
    redis_host: str = "localhost"
    redis_port: int = 6379

    model_config = settings_config


settings = Settings()


def get_encryption_key() -> str:
    encryption_key = settings.encryption_key
    if encryption_key is None or not (raw_key := encryption_key.get_secret_value()):
        raise KeyError("Encryption key is missing or empty")
    return raw_key
