<img src="docs/images/basejump-light.svg" alt="Basejump logo" width="400">


Basejump enables you to not only chat with your database using AI, but automates the process of setting up the process of building an AI-powered web app.

The following is built-in to Basejump to help you get started quickly:
- 🔒 Security: Ensures users and AI agents can only access the information they are provisioned
- ⚡ Indexing: Supports Redis as a vector database for fast database indexing
- 🗄️ Database: Sets up a schema to track chat history, clients, teams, users, and result history
- 💾 Caching: Support semantic caching for retrieval of datasets based on similar questions
- 📦 Saving: Saves data results in AWS S3 for later reference
- ✅ Accuracy: Basejump uses SQLglot to parse your database and ensure there are no hallucinated tables, columns, or where clause filters

# Installation

Create a virtual environment and then install from PYPI:
```
pip install basejump-core
```

Use the getting started section to index your database and prompt the AI with your first question.

# Getting Started
To get started using Basejump, there is a basejump-demo project to help you get started quickly.

## Dependencies

To run the project, you will need the following:
- An AWS account with access to S3 object storage
- Either AWS or Azure account with access to LLM AI models
- Docker installed on your computer

## Steps

Follow the following steps to get the demo project set up:
1. Change the .env_example file name to .env and fill out your credentials.
    - Some values have already been filled in as defaults, feel free to change these if you want.
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate the virtual environment. On a Mac, run this command: `source .venv/bin/activate`
4. Install `pip install dotenv-cli`
5. Change your directory to the basejump demo: `cd basejump-demo`
6. Run docker compose to spin up your containers: `dotenv docker compose up -d`
7. Run the main.py file: `docker-compose exec app python main.py`

After completing these steps, you should see the AI respond to your question based on the basejump database schema.

## Next steps

### Index your own database

Update the client database to your own test database to explore how Basejump AI is able to handle your own data.
Modify this line here: https://github.com/basejump-ai/basejump/blob/ad0a4aa569a57e38f8343fb509d530ba4a0a9652/basejump-demo/main.py#L87

### Create a free account on Basejump AI

If you want to see the `basejump-core` open-source project in action, you can check out https://basejump.ai/ to see how we implemented it.
Docs on using the web interface can be found here: https://docs.basejump.ai/

### Check out the open Basejump API

The Basejump API can be found here: https://docs.basejump.ai/api/api-reference

# Related Projects
Basejump would not be possible without all of the open-source projects it is built on. The following are vital to the success of Basejump:
- [Llama Index](https://github.com/run-llama/llama_index): For AI Data Agents
- [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy): For database inspection and connections
- [SQLGlot](https://github.com/tobymao/sqlglot): SQL query parsing

# Supported Databases
The following databases are currently supported. If you don't see one, submit a PR to get yours added:
- Postgres
- Snowflake
- Athena
- MySQL
- Redshift
- SQL Server

# Supported AI Models
Basejump is built on Llama Index and can support any AI models Llama Index supports. However, since there is a delay in our latest version and Llama Index, there are less models supported here. 
Adding support for a new model is relatively straightforward though, so please request one if you don't see it. Basejump AI currently only supports non-reasoning models, but reasoning models will be added soon!
- GPT4o
- GPT4.1
- Sonnet 3.5
- Sonnet 3.7




