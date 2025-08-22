"""
Configuration module for MCP News Agent
Handles loading and validation of environment variables and settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class MCPConfig:
    """Configuration settings for MCP News Agent"""
    
    def __init__(self):
        # API Keys
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY", "")
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "real-time-news-data.p.rapidapi.com")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        
        # MCP Server Settings
        self.mcp_server_host = os.getenv("MCP_SERVER_HOST", "localhost")
        self.mcp_server_port = int(os.getenv("MCP_SERVER_PORT", "3000"))
        
        # News Tool Settings
        self.default_language = os.getenv("DEFAULT_LANGUAGE", "en")
        self.default_country = os.getenv("DEFAULT_COUNTRY", "US")
        self.max_news_articles = int(os.getenv("MAX_NEWS_ARTICLES", "10"))
        self.max_summary_length = int(os.getenv("MAX_SUMMARY_LENGTH", "500"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")


# Global configuration instance
config = MCPConfig()


def validate_config():
    """Validate that required configuration is present"""
    missing_keys = []
    
    if not config.rapidapi_key:
        missing_keys.append("RAPIDAPI_KEY")
    
    if not config.mistral_api_key:
        missing_keys.append("MISTRAL_API_KEY")
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    return True
