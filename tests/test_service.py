from basejump.core.service.agents.utils import parse_message


def test_message_parsing():
    text = """
    Here’s my plan for your request:

    >> Identify the filters for the query based on the initial user prompt: Claims related to type 2 diabetes
    - Filter for claims that are associated with the chronic condition “Type 2 Diabetes.”

    >> Determine if you have enough information or if you need to ask the user clarifying questions.
    - The filter “Type 2 Diabetes” is clear and matches the chronic condition value set.

    I will now retrieve claims related to type 2 diabetes, including claim ID, person ID, claim start date, claim type, payer, plan, paid amount, and charge amount. If you want additional details or a specific time frame, let me know!
    """  # noqa

    message_result = parse_message(text=text)
    message = "\n\n".join(message_result)
    assert (
        message
        == "Here’s my plan for your request:\n\nI will now retrieve claims related to type 2 diabetes, including claim ID, person ID, claim start date, claim type, payer, plan, paid amount, and charge amount. If you want additional details or a specific time frame, let me know!"  # noqa
    )
    text = """
    Here’s my plan for your request:
    >> Identify the filters for the query based on the initial user prompt: Claims related to type 2 diabetes
    - Filter for claims that are associated with the chronic condition “Type 2 Diabetes.”

    I will now retrieve claims related to type 2 diabetes.
    >> Determine if you have enough information or if you need to ask the user clarifying questions.
    - The filter “Type 2 Diabetes” is clear and matches the chronic condition value set.

    This will include claim ID, person ID, claim start date, claim type, payer, plan, paid amount, and charge amount. If you want additional details or a specific time frame, let me know!
    """  # noqa
    message_result = parse_message(text=text)
    message = "\n\n".join(message_result)
    assert (
        message
        == """Here’s my plan for your request:\n\nI will now retrieve claims related to type 2 diabetes.\n\nThis will include claim ID, person ID, claim start date, claim type, payer, plan, paid amount, and charge amount. If you want additional details or a specific time frame, let me know!"""  # noqa
    )


test_message_parsing()
