@echo off
echo Starting MCP News Analysis Server...
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Virtual environment not found. Please run setup.py first.
    pause
    exit /b 1
)

REM Activate virtual environment and start server
call .venv\Scripts\Activate.bat
echo Virtual environment activated
echo.

echo Starting simple MCP server for testing...
python server\mcp_server.py

pause
