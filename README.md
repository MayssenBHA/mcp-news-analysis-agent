# mcp-news-analysis-agent

A comprehensive Model Context Protocol (MCP) implementation for intelligent news analysis, featuring advanced LLM-powered sentiment analysis, AI summarization, and natural language query processing using Mistral AI.

## 🧠 LLM-Powered Architecture

This project leverages **Mistral AI** for advanced natural language processing capabilities:

- **Sentiment Analysis**: Uses Mistral's LLM for nuanced sentiment understanding with confidence scoring
- **Text Summarization**: AI-powered content summarization with customizable length and style
- **Intent Detection**: Smart query interpretation for natural language interaction
- **Structured Outputs**: JSON-formatted responses with detailed reasoning and metadata

## 🚀 Features

- **News Fetching**: Retrieve real-time news articles from RapidAPI
- **Sentiment Analysis**: Advanced sentiment analysis using Mistral AI with confidence scoring and detailed reasoning
- **Text Summarization**: AI-powered summarization using Mistral AI
- **Intelligent Agent**: Natural language query processing with enhanced intent detection
- **MCP Architecture**: Fully compliant with Model Context Protocol standards

## 📋 Project Structure

```
MCPDemo/
├── server/
│   └── mcp_server.py          # MCP server implementation
├── client/
│   └── mcp_client.py          # Enhanced MCP client with intelligent intent detection and quote parsing
├── tools/
│   ├── news_tool.py           # News fetching from RapidAPI
│   ├── sentiment_tool.py      # sentiment analysis tools
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
# Ensure Mistral AI client is properly installed
pip install mistralai

# If you encounter any import errors, install packages individually:
pip install httpx langchain-mistralai fastmcp
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

# Run the client
python client/mcp_client.py
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
- "This new AI technology is amazing but also quite expensive" (direct text analysis)
```

## 🛠 Available Tools

### 1. News Tool
- **Function**: `fetch_news`
- **Purpose**: Fetches news articles from RapidAPI
- **Parameters**: topic, country, language, limit
- **Example**: Retrieve tech news from US in English

### 2. Sentiment Analysis Tool
- **Function**: `analyze_sentiment`
- **Purpose**: Analyzes sentiment using advanced Mistral AI LLM with confidence scoring and detailed reasoning
- **Parameters**: text, analysis_type (simple/detailed)
- **Features**: 
  - Structured JSON responses with confidence scores
  - Detailed reasoning and emotion detection
  - Support for complex, nuanced sentiment analysis
  - Direct text analysis through quotes
- **Example**: Determine sentiment with confidence: "Mixed sentiment (0.80 confidence) - expresses both excitement and concern"

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

## 🎯 Advanced Features

### Intelligent Text Detection
The client automatically detects quoted text in user queries and analyzes it directly:
- Input: `"This product is amazing but expensive"`
- Result: Direct sentiment analysis of the quoted text

### Structured LLM Responses
All LLM operations return structured JSON with:
- **Classification**: Primary sentiment/summary category
- **Confidence**: Numerical confidence score (0.0-1.0)
- **Reasoning**: Detailed explanation of the analysis
- **Emotions**: Additional emotional context (for detailed analysis)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed and virtual environment is activated
2. **Mistral API Key Errors**: Verify your Mistral AI API key is correctly set in `.env` file
3. **RapidAPI Errors**: Check if the provided RapidAPI key is still valid
4. **MCP Connection Issues**: Ensure both server and client are using the same transport method
5. **LLM Response Issues**: Verify Mistral AI API connectivity and sufficient API credits

### Checking Logs

The system uses Python logging. Check console output for detailed error messages. You can adjust log level in `.env`:

```env
LOG_LEVEL=DEBUG  # For more detailed logs
```

### Testing Individual Components

Test each tool separately:

```python
# Test LLM-powered sentiment analysis
from tools.sentiment_tool import SentimentTool
import asyncio

async def test_sentiment():
    tool = SentimentTool()
    result = await tool.analyze_sentiment(
        "This new AI technology is amazing but also quite expensive",
        "detailed"
    )
    print(result)

asyncio.run(test_sentiment())
```

## 📚 Dependencies

### Core MCP Dependencies
- `mcp>=1.2.0` - Model Context Protocol implementation
- `httpx>=0.25.0` - HTTP client for API requests
- `python-dotenv>=1.0.0` - Environment variable management

### AI/ML Dependencies
- `mistralai>=1.0.0` - Mistral AI client for both summarization and sentiment analysis
- `langchain-mistralai>=0.1.0` - LangChain integration for enhanced LLM capabilities
- `fastmcp>=2.11.0` - FastMCP framework for efficient MCP implementation

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
