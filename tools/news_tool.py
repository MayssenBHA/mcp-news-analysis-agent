"""
News Tool - Fetches news articles from RapidAPI
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from config.settings import config


class NewsTool:
    """Tool for fetching news articles using RapidAPI"""
    
    def __init__(self):
        self.base_url = f"https://{config.rapidapi_host}"
        self.headers = {
            'x-rapidapi-host': config.rapidapi_host,
            'x-rapidapi-key': config.rapidapi_key
        }
        
        # Country code mappings for better UX
        self.country_mappings = {
            'UK': 'GB',  # United Kingdom -> Great Britain
            'USA': 'US', # United States of America -> US
            'ENGLAND': 'GB',
            'BRITAIN': 'GB'
        }
    
    async def fetch_news(
        self, 
        topic: Optional[str] = None,
        country: str = config.default_country,
        language: str = config.default_language,
        limit: int = config.max_news_articles
    ) -> Dict[str, Any]:
        """
        Fetch news articles based on topic, country, and language
        
        Args:
            topic: News topic to search for
            country: Country code (e.g., 'US', 'GB', 'FR') - UK will be converted to GB
            language: Language code (e.g., 'en', 'fr', 'es')
            limit: Maximum number of articles to return
            
        Returns:
            Dictionary containing news articles and metadata
        """
        try:
            # Convert common country names to proper codes
            country = self.country_mappings.get(country.upper(), country)
            
            async with httpx.AsyncClient() as client:
                # Build the API endpoint URL
                if topic:
                    url = f"{self.base_url}/search"
                    params = {
                        'query': topic,
                        'country': country,
                        'lang': language,
                        'limit': limit
                    }
                else:
                    url = f"{self.base_url}/top-headlines"
                    params = {
                        'country': country,
                        'lang': language,
                        'limit': limit
                    }
                
                response = await client.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'articles': self._process_articles(data.get('data', [])),
                        'total_results': len(data.get('data', [])),
                        'query': topic or 'top headlines',
                        'timestamp': datetime.now().isoformat()
                    }
                elif response.status_code == 429:
                    return {
                        'success': False,
                        'error': "Rate limit exceeded. Please wait before making more requests.",
                        'message': response.text
                    }
                elif response.status_code == 400:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', 'Bad request')
                    return {
                        'success': False,
                        'error': f"API error: {error_msg}",
                        'message': response.text
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API request failed with status {response.status_code}",
                        'message': response.text
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to fetch news: {str(e)}",
                'articles': []
            }
    
    async def fetch_story_coverage(self, story_id: str) -> Dict[str, Any]:
        """
        Fetch full story coverage using the provided example endpoint
        
        Args:
            story_id: Story identifier
            
        Returns:
            Dictionary containing story coverage data
        """
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/full-story-coverage"
                params = {
                    'story': story_id,
                    'sort': 'RELEVANCE',
                    'country': config.default_country,
                    'lang': config.default_language
                }
                
                response = await client.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'coverage': data,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API request failed with status {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to fetch story coverage: {str(e)}"
            }
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Process and clean article data
        
        Args:
            articles: Raw article data from API
            
        Returns:
            List of processed articles
        """
        processed = []
        
        for article in articles[:config.max_news_articles]:
            processed_article = {
                'title': article.get('title', 'No title'),
                'description': article.get('snippet', article.get('description', 'No description')),
                'url': article.get('link', article.get('url', '')),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('published_datetime_utc', ''),
                'author': article.get('author', 'Unknown'),
                'category': article.get('category', 'General'),
                'content': article.get('content', '')[:1000] if article.get('content') else ''  # Limit content length
            }
            processed.append(processed_article)
        
        return processed


# Tool registration function for MCP server
async def get_news_tool_definition():
    """Return the tool definition for MCP server registration"""
    return {
        "name": "fetch_news",
        "description": "Fetch news articles by topic, country, and language from real-time news sources",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "News topic to search for (optional)"
                },
                "country": {
                    "type": "string", 
                    "description": "Country code (e.g., 'US', 'UK', 'FR')",
                    "default": config.default_country
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g., 'en', 'fr', 'es')",
                    "default": config.default_language
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of articles to return",
                    "default": config.max_news_articles,
                    "minimum": 1,
                    "maximum": 50
                }
            }
        }
    }


async def execute_news_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the news tool with given arguments"""
    news_tool = NewsTool()
    return await news_tool.fetch_news(
        topic=args.get('topic'),
        country=args.get('country', config.default_country),
        language=args.get('language', config.default_language),
        limit=args.get('limit', config.max_news_articles)
    )
