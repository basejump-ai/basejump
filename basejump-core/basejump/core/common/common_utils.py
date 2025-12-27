"""Utility functions that aren't dependent on any other basejump module"""

import hashlib
import re
from datetime import datetime


def get_current_datetime():
    return datetime.now().replace(microsecond=0)


def hash_value(value: str):
    encoded_value = value.encode("UTF-8")
    hashed_value = hashlib.sha256(encoded_value).hexdigest()
    return hashed_value


def is_valid_email(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[a-zA-Z\d-]+\.[a-zA-Z]{2,}$"
    return re.fullmatch(pattern, email) is not None
