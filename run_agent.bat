@echo off
echo ✅ Starting MCP News Analysis Agent with proper MCP client-server connection...
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo ❌ Virtual environment not found. Please run setup.py first.
    pause
    exit /b 1
)

REM Activate virtual environment and start agent
call .venv\Scripts\Activate.bat
echo ✅ Virtual environment activated
echo.

echo 🔄 Starting MCP agent with FastMCP server connection...
echo 💡 The agent connects to the MCP server and uses MCP tools
echo 📡 Type 'help' for example queries, 'quit' to exit
echo.
python client\mcp_client.py

pause
