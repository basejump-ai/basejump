# Security Policy

## Security Considerations
`basejump` is a Python library intended to be used inside a trusted execution environment (e.g., server-side backend code, scheduled jobs, local scripts). When you embed `basejump` in a network-facing service (FastAPI, Flask, Django, etc.), you are creating a new attack surface that is outside the scope of this project.

This library provides direct access to database querying capabilities. **You are responsible for**:
* Validation and sanitization of all user-supplied input (including prompts, database credentials, connection parameters, and SQL queries) is the responsibility of the hosting application.
* Authentication, authorization, rate limiting, and other common web security controls are similarly out of scope.
* `basejump` assumes that the caller has already performed any necessary filtering, validation, and access control checks before data reaches the library.

Basejump provides access control features (clients, teams, users), but configuring and enforcing these policies is your responsibility.

## Reporting a Vulnerability
Please report security vulnerabilities to product@basejump.ai