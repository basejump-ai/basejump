"""Utility functions that aren't dependent on any other basejump module"""

import hashlib
import os
from datetime import datetime


def get_current_datetime():
    return datetime.now().replace(microsecond=0)


def hash_value(value: str):
    encoded_value = value.encode("UTF-8")
    hashed_value = hashlib.sha256(encoded_value).hexdigest()
    return hashed_value


def find_markdown_files(file_path: str):
    markdown_files = []

    # Walk through the directory
    for dirpath, _, filenames in os.walk(file_path):
        for filename in filenames:
            if filename.endswith(".md"):
                # Append the full path of the markdown file
                markdown_files.append(os.path.join(dirpath, filename))

    return markdown_files
