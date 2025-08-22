"""
Summary Tool - Summarizes long news articles and text content
"""

from typing import Dict, Any, List, Optional
import asyncio
import re
from mistralai import Mistral
from config.settings import config


class SummaryTool:
    """Tool for summarizing text content using Mistral AI"""
    
    def __init__(self):
        if not config.mistral_api_key:
            raise ValueError("Mistral API key is required. Please set MISTRAL_API_KEY in your .env file")
        
        self.client = Mistral(api_key=config.mistral_api_key)
        self.model = "mistral-small-latest"  # Using a reliable model
    
    async def summarize_text(
        self, 
        text: str, 
        max_length: int = config.max_summary_length,
        summary_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Summarize given text using Mistral AI
        
        Args:
            text: Text content to summarize
            max_length: Maximum length of summary in characters
            summary_type: Type of summary ('brief', 'comprehensive', 'bullet_points')
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'No text provided for summarization'
            }
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        if len(cleaned_text) < 100:
            return {
                'success': True,
                'original_text': cleaned_text,
                'summary': cleaned_text,
                'compression_ratio': 1.0,
                'message': 'Text too short to summarize meaningfully'
            }
        
        try:
            # Create appropriate prompt based on summary type
            prompt = self._create_summary_prompt(cleaned_text, max_length, summary_type)
            
            # Call Mistral AI API
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=min(max_length // 2, 500),  # Estimate token count
                temperature=0.3  # Lower temperature for more consistent summaries
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'original_length': len(cleaned_text),
                'summary_length': len(summary),
                'summary': summary,
                'compression_ratio': round(len(summary) / len(cleaned_text), 3),
                'summary_type': summary_type,
                'model_used': self.model
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Summarization failed: {str(e)}',
                'original_length': len(cleaned_text)
            }
    
    async def summarize_articles(
        self, 
        articles: List[Dict[str, Any]], 
        individual_summaries: bool = True,
        collective_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize multiple news articles
        
        Args:
            articles: List of article dictionaries
            individual_summaries: Whether to create individual summaries
            collective_summary: Whether to create a collective summary
            
        Returns:
            Dictionary containing all summaries and metadata
        """
        if not articles:
            return {
                'success': False,
                'error': 'No articles provided for summarization'
            }
        
        try:
            result = {'success': True, 'total_articles': len(articles)}
            
            # Individual article summaries
            if individual_summaries:
                individual_results = []
                
                for i, article in enumerate(articles):
                    # Combine title, description, and content
                    full_text = self._combine_article_text(article)
                    
                    summary_result = await self.summarize_text(
                        full_text, 
                        max_length=200,  # Shorter summaries for individual articles
                        summary_type="brief"
                    )
                    
                    individual_results.append({
                        'article_index': i,
                        'title': article.get('title', 'No title')[:100],
                        'source': article.get('source', 'Unknown'),
                        'summary_result': summary_result
                    })
                
                result['individual_summaries'] = individual_results
            
            # Collective summary
            if collective_summary:
                # Combine all articles for collective summary
                all_articles_text = "\n\n---\n\n".join([
                    self._combine_article_text(article) for article in articles
                ])
                
                collective_result = await self.summarize_text(
                    all_articles_text,
                    max_length=config.max_summary_length,
                    summary_type="comprehensive"
                )
                
                result['collective_summary'] = collective_result
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Batch summarization failed: {str(e)}'
            }
    
    async def create_news_briefing(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a structured news briefing from multiple articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary containing structured news briefing
        """
        if not articles:
            return {
                'success': False,
                'error': 'No articles provided for news briefing'
            }
        
        try:
            # Prepare articles text with structure
            briefing_text = "Create a structured news briefing from the following articles:\n\n"
            
            for i, article in enumerate(articles[:5]):  # Limit to 5 articles to avoid token limits
                briefing_text += f"Article {i+1}:\n"
                briefing_text += f"Title: {article.get('title', 'No title')}\n"
                briefing_text += f"Source: {article.get('source', 'Unknown')}\n"
                briefing_text += f"Content: {self._combine_article_text(article)[:500]}...\n\n"
            
            prompt = f"""{briefing_text}

Please create a professional news briefing with the following structure:
1. Executive Summary (2-3 sentences)
2. Key Headlines (bullet points)
3. Main Stories (brief summaries)
4. Overall Themes/Trends
5. Key Takeaways

Keep the briefing concise but informative, suitable for a busy executive."""
            
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=800,
                temperature=0.3
            )
            
            briefing = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'briefing': briefing,
                'articles_included': len(articles[:5]),
                'total_articles_available': len(articles),
                'model_used': self.model
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'News briefing creation failed: {str(e)}'
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for summarization"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text.strip()
    
    def _combine_article_text(self, article: Dict[str, Any]) -> str:
        """Combine article title, description, and content into single text"""
        parts = []
        
        if article.get('title'):
            parts.append(f"Title: {article['title']}")
        
        if article.get('description'):
            parts.append(f"Description: {article['description']}")
        
        if article.get('content'):
            parts.append(f"Content: {article['content']}")
        
        return "\n".join(parts)
    
    def _create_summary_prompt(self, text: str, max_length: int, summary_type: str) -> str:
        """Create appropriate prompt for summarization based on type"""
        base_prompt = f"Please summarize the following text"
        
        if summary_type == "brief":
            prompt = f"{base_prompt} in a brief, concise manner (maximum {max_length} characters):\n\n{text}"
        elif summary_type == "bullet_points":
            prompt = f"{base_prompt} as key bullet points (maximum {max_length} characters):\n\n{text}"
        else:  # comprehensive
            prompt = f"{base_prompt} comprehensively but concisely (maximum {max_length} characters):\n\n{text}"
        
        return prompt


# Tool registration functions for MCP server
async def get_summary_tool_definition():
    """Return the summary tool definition for MCP server registration"""
    return {
        "name": "summarize_text",
        "description": "Summarize text content using Mistral AI with different summary types",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to summarize"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length of summary in characters",
                    "default": config.max_summary_length,
                    "minimum": 50,
                    "maximum": 2000
                },
                "summary_type": {
                    "type": "string",
                    "description": "Type of summary to generate",
                    "enum": ["brief", "comprehensive", "bullet_points"],
                    "default": "comprehensive"
                }
            },
            "required": ["text"]
        }
    }


async def get_articles_summary_tool_definition():
    """Return the articles summary tool definition for MCP server registration"""
    return {
        "name": "summarize_articles",
        "description": "Summarize multiple news articles individually and/or collectively",
        "inputSchema": {
            "type": "object",
            "properties": {
                "articles": {
                    "type": "array",
                    "description": "Array of news articles to summarize",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "content": {"type": "string"},
                            "source": {"type": "string"}
                        }
                    }
                },
                "individual_summaries": {
                    "type": "boolean",
                    "description": "Whether to create individual article summaries",
                    "default": True
                },
                "collective_summary": {
                    "type": "boolean",
                    "description": "Whether to create a collective summary of all articles",
                    "default": True
                }
            },
            "required": ["articles"]
        }
    }


async def get_news_briefing_tool_definition():
    """Return the news briefing tool definition for MCP server registration"""
    return {
        "name": "create_news_briefing",
        "description": "Create a structured professional news briefing from multiple articles",
        "inputSchema": {
            "type": "object",
            "properties": {
                "articles": {
                    "type": "array",
                    "description": "Array of news articles for the briefing",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "content": {"type": "string"},
                            "source": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["articles"]
        }
    }


async def execute_summary_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the summary tool with given arguments"""
    summary_tool = SummaryTool()
    return await summary_tool.summarize_text(
        text=args.get('text', ''),
        max_length=args.get('max_length', config.max_summary_length),
        summary_type=args.get('summary_type', 'comprehensive')
    )


async def execute_articles_summary_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the articles summary tool with given arguments"""
    summary_tool = SummaryTool()
    return await summary_tool.summarize_articles(
        articles=args.get('articles', []),
        individual_summaries=args.get('individual_summaries', True),
        collective_summary=args.get('collective_summary', True)
    )


async def execute_news_briefing_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the news briefing tool with given arguments"""
    summary_tool = SummaryTool()
    return await summary_tool.create_news_briefing(
        articles=args.get('articles', [])
    )
