import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from core.gemini import Gemini  # We'll keep this for compatibility
from mcp_client import MCPClient
from core.gemini import Gemini  # We'll create this instead of Claude
from core.cli_chat import CliChat
from core.cli import CliApp
from core.gemini import Gemini

# ...


load_dotenv()

# Gemini Config
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Default free model
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

gemini_service = Gemini(model=gemini_model, api_key=gemini_api_key)

assert gemini_model, "Error: GEMINI_MODEL cannot be empty. Update .env"
assert gemini_api_key, "Error: GEMINI_API_KEY cannot be empty. Update .env"


async def main():
    # Use Gemini instead of Claude
    gemini_service = Gemini(model=gemini_model, api_key=gemini_api_key)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            gemini_service=Gemini(model=gemini_model, api_key=gemini_api_key),  # Still named claude_service for compatibility
        )
### *****
        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
