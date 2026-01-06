"""Utility functions that aren't dependent on any other basejump module"""

import hashlib
from datetime import datetime


def get_current_datetime():
    return datetime.now().replace(microsecond=0)


def hash_value(value: str):
    encoded_value = value.encode("UTF-8")
    hashed_value = hashlib.sha256(encoded_value).hexdigest()
    return hashed_value
