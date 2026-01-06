This package contains basejump-core as well as basejump[dev]. It's a convenience package to quickly import all basejump packages.

Basejump indexes a database and connects it with an AI data agent to chat with your data.

## Key Features
* ✅ **Accuracy**: Uses SQLglot to parse and validate queries, preventing hallucinated tables, columns, or filters
* 🔒 **Security**: Role-based access control ensures users and AI agents only access provisioned data
* ⚡ **Fast Indexing**: Redis vector database integration for rapid semantic search
* 🗄️ **Full Tracking**: Pre-configured schema tracks chat history, clients, teams, users, and query results
- 💾 **Smart Caching**: Support semantic caching for retrieval of datasets based on similar questions
- 📦 **Result Storage**: Saves data results for later reference and auditing

## Installation
Create a virtual environment and then install from PYPI:
```bash
pip install basejump-core
```

### Example usage
A complete working example can be found in the [basejump repo](https://github.com/basejump-ai/basejump) under `basejump-demo/main.py`. Here's the core functionality in just 10 lines:

```python
async with service.run_session() as (core_session, db):
    service_context = service.create_service_context(core_session)
    user_info = await service.create_internal_user_info(db, service_context)
    connection = await service.setup_database(db, service_context, user_info, client_conn_params)
    await service.chat(
        db,
        "Provide a report of all clients.",
        service_context,
        user_info,
        connection,
    )
```

## Next steps

### Index your own database
Modify the `client_conn_params` in basejump-demo/settings to your own test database to explore how Basejump AI responds to your data.

### Try Basejump Cloud
If you want to see the basejump open source project in action, you can check out https://basejump.ai/ to see how we implemented it.
Docs on using the web interface can be found here: https://docs.basejump.ai/

The Basejump API docs can be found here: https://docs.basejump.ai/api/api-reference

## Related Projects
Basejump would not be possible without all of the open source projects it is built on. The following are vital to the success of Basejump :clap:
- [Llama Index](https://github.com/run-llama/llama_index): For AI data agents
- [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy): For database inspection and connections
- [SQLGlot](https://github.com/tobymao/sqlglot): SQL query parsing

## Supported Databases
The following databases are currently supported. If you don't see one, submit a PR to get yours added:
- Postgres
- Snowflake
- Athena
- MySQL
- Redshift
- SQL Server

## Supported AI Models
Basejump is built on Llama Index and can theoretically support any AI models Llama Index supports.
Adding support for a new model is relatively straightforward, so please request one if you don't see it.

Most of the LLMS from OpenAI and Anthropic are available via Azure and AWS respectively. Supported Claude models can be found [here](https://github.com/basejump-ai/basejump-llama-index/blob/main/llama-index-integrations/llms/llama-index-llms-bedrock-converse/llama_index/llms/bedrock_converse/utils.py). Supported OpenAI models can be found [here](https://github.com/basejump-ai/basejump-llama-index/blob/main/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/utils.py).

To add a new model to Basejump, just update the `AIModelSchema` in `basejump.core.models.schemas` and submit a PR.
