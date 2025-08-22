# mcp-news-analysis-agent

A comprehensive Model Context Protocol (MCP) implementation for news analysis, featuring sentiment analysis, summarization, and intelligent query processing.

## 🚀 Features

- **News Fetching**: Retrieve real-time news articles from RapidAPI
- **Sentiment Analysis**: Analyze sentiment using TextBlob and VADER
- **Text Summarization**: AI-powered summarization using Mistral AI
- **Intelligent Agent**: Natural language query processing
- **MCP Architecture**: Fully compliant with Model Context Protocol standards

## 📋 Project Structure

```
MCPDemo/
├── server/
│   └── mcp_server.py          # MCP server implementation
├── client/
│   └── mcp_agent.py           # Intelligent news analysis agent
├── tools/
│   ├── news_tool.py           # News fetching from RapidAPI
│   ├── sentiment_tool.py      # Sentiment analysis tools
│   ├── summary_tool.py        # Text summarization tools
│   └── __init__.py
├── config/
│   ├── settings.py            # Configuration management
│   ├── .env                   # Environment variables
│   └── __init__.py
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Setup Instructions

### 1. Environment Setup

First, create a Python virtual environment:

```powershell
# Navigate to project directory
cd "C:\Users\mayssen\Desktop\mcp project\MCPDemo"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Configuration

Edit the `config/.env` file and add your API keys:

```env
# News API key from RapidAPI (already provided)
RAPIDAPI_KEY=6d35e9aa82msh4c8550ffb3e08b4p15bf78jsna3f5a47eeb4d
RAPIDAPI_HOST=real-time-news-data.p.rapidapi.com

# Get your Mistral AI API key from https://console.mistral.ai/
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional: Adjust other settings as needed
MAX_NEWS_ARTICLES=10
MAX_SUMMARY_LENGTH=500
LOG_LEVEL=INFO
```

### 3. Install Additional Dependencies

Some packages might need special installation:

```powershell
# Install NLTK data for text processing
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# If you encounter any import errors, install packages individually:
pip install textblob vaderSentiment mistralai
```

## 🔧 Usage

### Running the MCP Server

```powershell
# Make sure you're in the project directory with activated virtual environment
python server/mcp_server.py
```

### Running the Interactive Agent

In a separate terminal:

```powershell
# Activate the same virtual environment
.\.venv\Scripts\Activate

# Run the agent
python client/mcp_agent.py
```

### Example Queries

Once the agent is running, try these natural language queries:

```
- "Get latest news about technology"
- "Analyze sentiment of recent news about climate change"
- "Summarize news about the economy"
- "Show me top 5 news from UK"
- "How do people feel about the latest political news?"
- "Get French news about sports and analyze sentiment"
```

## 🛠 Available Tools

### 1. News Tool
- **Function**: `fetch_news`
- **Purpose**: Fetches news articles from RapidAPI
- **Parameters**: topic, country, language, limit
- **Example**: Retrieve tech news from US in English

### 2. Sentiment Analysis Tool
- **Function**: `analyze_sentiment`
- **Purpose**: Analyzes sentiment using TextBlob and VADER
- **Parameters**: text, method (textblob/vader/comprehensive)
- **Example**: Determine if news coverage is positive or negative

### 3. Summary Tool (Requires Mistral AI)
- **Function**: `summarize_text`
- **Purpose**: Summarizes text using Mistral AI
- **Parameters**: text, max_length, summary_type
- **Example**: Create brief summaries of long articles

### 4. Combined Workflows
- **Function**: `analyze_news_sentiment`
- **Purpose**: Fetches news and analyzes sentiment in one step
- **Parameters**: topic, country, language, limit
- **Example**: Get tech news and determine public sentiment

## 🔌 Integration with Claude Desktop

To use this server with Claude Desktop, add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "news-analysis": {
      "command": "python",
      "args": ["C:/Users/mayssen/Desktop/mcp project/MCPDemo/server/mcp_server.py"],
      "env": {
        "PYTHONPATH": "C:/Users/mayssen/Desktop/mcp project/MCPDemo"
      }
    }
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed and virtual environment is activated
2. **API Key Errors**: Verify your Mistral API key is correctly set in `.env`
3. **RapidAPI Errors**: Check if the provided RapidAPI key is still valid
4. **MCP Connection Issues**: Ensure both server and client are using the same transport method

### Checking Logs

The system uses Python logging. Check console output for detailed error messages. You can adjust log level in `.env`:

```env
LOG_LEVEL=DEBUG  # For more detailed logs
```

### Testing Individual Components

Test each tool separately:

```python
# Test news fetching
from tools.news_tool import NewsTool
import asyncio

async def test_news():
    tool = NewsTool()
    result = await tool.fetch_news("technology", "US", "en", 5)
    print(result)

asyncio.run(test_news())
```

## 📚 Dependencies

### Core MCP Dependencies
- `mcp>=1.2.0` - Model Context Protocol implementation
- `httpx>=0.25.0` - HTTP client for API requests
- `python-dotenv>=1.0.0` - Environment variable management

### AI/ML Dependencies
- `mistralai>=1.0.0` - Mistral AI client for summarization
- `textblob>=0.17.1` - Text sentiment analysis
- `vaderSentiment>=3.3.2` - VADER sentiment analyzer

### Utility Dependencies
- `requests>=2.31.0` - HTTP requests
- `pydantic>=2.0.0` - Data validation
- `rich>=13.0.0` - Pretty terminal output

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review logs for error messages
3. Verify API keys and configuration
4. Test individual components

For additional help, review the MCP documentation at [Model Context Protocol](https://modelcontextprotocol.io/).
>>>>>>> 76537c4 (initial commit)
