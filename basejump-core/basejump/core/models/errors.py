"""Used for constants, schemas, and functions related to error handling"""

INVALID_SCHEMAS = """Only permitted schemas can be provided. \
"The following non-permitted schemas were passed in: {non_perm_schemas}"""


class InvalidSchemas(Exception):
    def __init__(self, non_perm_schemas):
        msg = INVALID_SCHEMAS.format(non_perm_schemas=non_perm_schemas)
        super().__init__(msg)


class SSLError(Exception):
    """Base exception for SSL-related errors"""

    pass


SSL_CONFIG_ERROR = "Missing SSL root certificate. Need ssl root cert if ssl mode is greater than require."


class SSLConfigError(SSLError):
    def __init__(self):
        super().__init__(SSL_CONFIG_ERROR)


class LowConfidenceResponse(Exception):
    pass


MERMAID_REFRESH_ERROR = """The Mermaid diagram is unable to be refreshed due to issues with processing. \
If this persists, please reach out to Basejump support."""


class MermaidRefreshError(Exception):
    def __init__(self):
        super().__init__(MERMAID_REFRESH_ERROR)


INVALID_CLIENT_CREDENTIALS = """The AWS Access credentials provided do not have access to this resource please double \
check your AWS Access Key, AWS Secret Key and Bucket name."""


class InvalidClientCredentials(Exception):
    def __init__(self):
        super().__init__(INVALID_CLIENT_CREDENTIALS)


class StrictModeFlagged(Exception):
    def __init__(self):
        super().__init__(
            "I was not able to find an answer to your solution when searching previous answers generated \
by admins. This is due to Verify Mode currently being set to STRICT. Please talk with your admin/data team to \
explore the data further and for follow-up questions. "
        )


class UnauthorizedUserRole(Exception):
    def __init__(self):
        super().__init__("A higher level of access is required")


UNAUTHORIZED_USER_VERIFY_ROLE = """Your user role does not have a high enough permission level to change the \
verification level. You must be an {role_level} or higher to edit the verification of this result."""


class UnauthorizedUserVerifyRole(Exception):
    def __init__(self, role_level):
        msg = UNAUTHORIZED_USER_VERIFY_ROLE.format(role_level=role_level)
        super().__init__(msg)


class StarQueryError(Exception):
    def __init__(self):
        super().__init__(
            "You are not allowed to use the * character to select columns. Name the columns directly instead."
        )


class HallucinatedColumnError(Exception):
    pass


class ColumnCapitalizationError(Exception):
    pass


class SQLParseError(Exception):
    pass


class SQLRunError(Exception):
    pass


class NoRelevantTables(Exception):
    pass


class UnverifiedColumns(Exception):
    pass


class ConnectDBError(Exception):
    pass


class InactivePlanError(Exception):
    pass


class DoesNotExistError(Exception):
    pass


GET_TEAM_CONN_ERROR = "Chat not found based on the provided UUID"


class GetTeamConnError(Exception):
    def __init__(self):
        super().__init__(GET_TEAM_CONN_ERROR)


CHAT_UUID_NOT_FOUND = "Chat not found based on the provided UUID"


class ChatUUIDNotFound(Exception):
    def __init__(self):
        super().__init__(CHAT_UUID_NOT_FOUND)


GET_CHAT_HISTORY_ERROR = "Error retrieving chat history"


class GetChatHistoryError(Exception):
    def __init__(self):
        super().__init__(GET_CHAT_HISTORY_ERROR)


PROMPTING_AI_ERROR = (
    "Error prompting the AI. Please try again and reach out to Basejump support if the problem persists."
)


class PromptingAIError(Exception):
    def __init__(self):
        super().__init__(PROMPTING_AI_ERROR)


INVALID_SCHEMA_SYNTAX = """Invalid schema syntax either has more than 63 characters \
or has characters besides letters, numbers, or underscores"""
INVALID_SCHEMA_ARGS = "Either include_default_schema or schemas must be True."

INVALID_JINJA_BRACE_COUNT = "The number of opening and closing curly braces don't match"


class InvalidJinjaBraceCount(Exception):
    def __init__(self):
        super().__init__(INVALID_JINJA_BRACE_COUNT)


INVALID_JINJA_STARTING_BRACE = (
    "Opening curly brace found without curly brace immediately following to form double curly brace"
)


class InvalidJinjaStartingBrace(Exception):
    def __init__(self):
        super().__init__(INVALID_JINJA_STARTING_BRACE)


INVALID_JINJA_ENDING_BRACE = (
    "Closing curly brace found without curly brace immediately following to form double curly brace"
)


class InvalidJinjaEndingBrace(Exception):
    def __init__(self):
        super().__init__(INVALID_JINJA_ENDING_BRACE)


INVALID_JINJA_CONTENT = "Missing content after double jinja opening braces"


class InvalidJinjaContent(Exception):
    def __init__(self):
        super().__init__(INVALID_JINJA_CONTENT)


DB_ALIAS_CONFLICT = "This DB alias name already exists. Please pick a different DB alias name."


class DBAliasConflict(Exception):
    def __init__(self):
        super().__init__(DB_ALIAS_CONFLICT)


RESULT_UUID_NOT_FOUND = "Result not found based on the provided UUID"


class SQLIndexError(Exception):
    pass


class MissingJinjaKey(Exception):
    pass
