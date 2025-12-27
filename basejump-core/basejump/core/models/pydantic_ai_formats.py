from pydantic import BaseModel, Field


class MermaidJSFormat(BaseModel):
    """Data model to extract MermaidJS code"""

    mermaidjs_code: str = Field(description="The mermaidjs code that was generated.")


class TrueFalseBool(BaseModel):
    """Data model to extract True or False value"""

    true_false_bool: bool = Field(description="A True or False boolean value")


class CleanSQLFormat(BaseModel):
    """Data model to extract SQL queries."""

    sql_query: str = Field(
        description="""The sql query without any line breaks or unnecessary punctuation that would cause \
the SQL query to fail when executed."""
    )


class SubPrompts(BaseModel):
    """Data model to extract sub prompts"""

    sub_prompts: list = Field(
        description="A list of sub prompts where each item is extract for a number in a numbered list"
    )


class DescriptionFormat(BaseModel):
    """Data model to create title and descriptions for sql query results"""

    title: str = Field(description="Give the results a brief title.")
    subtitle: str = Field(
        description="""Give the result a 1 sentence subtitle that provides slightly more information than the title\
, but keep it brief."""
    )
    description: str = Field(description="A description of the results.")


class FormattedMetric(BaseModel):
    """Data model to create title and descriptions for sql query results"""

    metric_value: str = Field(description="Extract the relevant metric value")
    metric_value_formatted: str = Field(
        description="Take the provided metric and format the text to provide the appropriate context."
    )


class DateData(BaseModel):
    """A list of date strings formatted in the YYYY-MM-DD format"""

    dates: list[str] = Field(description="A list of date strings formatted in the YYYY-MM-DD format")
