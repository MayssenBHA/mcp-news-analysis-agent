"""
Sentiment Analysis Tool - AI-powered sentiment analysis using Mistral LLM
"""

from typing import Dict, Any, List
import asyncio
import json
import os
from mistralai import Mistral


class SentimentTool:
    """Tool for analyzing sentiment of text content using Mistral LLM"""
    
    def __init__(self):
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-tiny"
    
    async def analyze_sentiment(self, text: str, method: str = "llm") -> Dict[str, Any]:
        """
        Analyze sentiment of given text using Mistral LLM
        
        Args:
            text: Text to analyze
            method: Analysis method ('llm', 'detailed', or 'simple')
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'No text provided for sentiment analysis'
            }
        
        try:
            if method == "detailed":
                sentiment_result = await self._analyze_detailed_sentiment(text)
            else:
                sentiment_result = await self._analyze_simple_sentiment(text)
            
            return {
                'success': True,
                'text_length': len(text),
                'sentiment': sentiment_result,
                'analysis_method': 'mistral_llm'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'LLM sentiment analysis failed: {str(e)}'
            }
    
    async def _analyze_simple_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis using LLM"""
        prompt = f"""Analyze the sentiment of the following text and respond with ONLY a JSON object:

Text: "{text}"

Respond with this exact format:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content.strip())
            return {
                'classification': result['sentiment'],
                'confidence': float(result['confidence']),
                'reasoning': result['reasoning']
            }
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback if JSON parsing fails
            content = response.choices[0].message.content.lower()
            if 'positive' in content:
                classification = 'positive'
            elif 'negative' in content:
                classification = 'negative'
            else:
                classification = 'neutral'
            
            return {
                'classification': classification,
                'confidence': 0.7,
                'reasoning': 'Fallback analysis due to parsing error',
                'raw_response': response.choices[0].message.content
            }
    
    async def _analyze_detailed_sentiment(self, text: str) -> Dict[str, Any]:
        """Detailed sentiment analysis using LLM"""
        prompt = f"""Perform a comprehensive sentiment analysis of the following text. Respond with ONLY a JSON object:

Text: "{text}"

Analyze the sentiment across multiple dimensions and respond with this exact format:
{{
    "overall_sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.0-1.0,
    "emotional_tone": "happy|sad|angry|excited|calm|anxious|etc",
    "subjectivity": "objective|subjective|highly_subjective",
    "intensity": "low|medium|high",
    "key_phrases": ["phrase1", "phrase2"],
    "reasoning": "detailed explanation"
}}"""

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content.strip())
            return {
                'classification': result['overall_sentiment'],
                'confidence': float(result['confidence']),
                'emotional_tone': result['emotional_tone'],
                'subjectivity': result['subjectivity'],
                'intensity': result['intensity'],
                'key_phrases': result['key_phrases'],
                'reasoning': result['reasoning']
            }
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback analysis
            return await self._analyze_simple_sentiment(text)
    
    async def analyze_articles_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple news articles using LLM
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary containing aggregated sentiment analysis
        """
        if not articles:
            return {
                'success': False,
                'error': 'No articles provided for sentiment analysis'
            }
        
        try:
            article_sentiments = []
            
            for i, article in enumerate(articles):
                # Combine title and description for analysis
                text_to_analyze = f"{article.get('title', '')} {article.get('description', '')}"
                
                sentiment_result = await self.analyze_sentiment(text_to_analyze, method='llm')
                
                if sentiment_result['success']:
                    article_sentiment = {
                        'article_index': i,
                        'title': article.get('title', 'No title')[:100],
                        'sentiment_analysis': sentiment_result['sentiment']
                    }
                    article_sentiments.append(article_sentiment)
            
            # Calculate aggregate sentiment using LLM
            aggregate_sentiment = await self._calculate_aggregate_sentiment(article_sentiments)
            
            return {
                'success': True,
                'total_articles_analyzed': len(article_sentiments),
                'individual_sentiments': article_sentiments,
                'aggregate_sentiment': aggregate_sentiment
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Batch sentiment analysis failed: {str(e)}'
            }
    
    
    async def _calculate_aggregate_sentiment(self, article_sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate sentiment across multiple articles using LLM"""
        if not article_sentiments:
            return {'classification': 'neutral', 'confidence': 0.5}
        
        # Prepare summary of individual sentiments for LLM analysis
        sentiment_summary = []
        for article in article_sentiments:
            sentiment_data = article['sentiment_analysis']
            sentiment_summary.append({
                'title': article['title'],
                'sentiment': sentiment_data['classification'],
                'confidence': sentiment_data['confidence']
            })
        
        prompt = f"""Analyze the overall sentiment trend from these {len(sentiment_summary)} news articles. Respond with ONLY a JSON object:

Articles sentiment data:
{json.dumps(sentiment_summary, indent=2)}

Provide an aggregate analysis with this exact format:
{{
    "overall_sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.0-1.0,
    "sentiment_distribution": {{
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0
    }},
    "dominant_sentiment": "most common sentiment",
    "trend_analysis": "brief analysis of the overall trend"
}}"""

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return {
                'overall_sentiment': result['overall_sentiment'],
                'confidence': float(result['confidence']),
                'sentiment_distribution': result['sentiment_distribution'],
                'dominant_sentiment': result['dominant_sentiment'],
                'trend_analysis': result['trend_analysis']
            }
            
        except Exception as e:
            # Fallback to simple counting
            classifications = [article['sentiment_analysis']['classification'] for article in article_sentiments]
            positive_count = classifications.count('positive')
            negative_count = classifications.count('negative')
            neutral_count = classifications.count('neutral')
            
            dominant = max(['positive', 'negative', 'neutral'], key=classifications.count)
            
            return {
                'overall_sentiment': dominant,
                'confidence': 0.7,
                'sentiment_distribution': {
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count
                },
                'dominant_sentiment': dominant,
                'trend_analysis': f'Fallback analysis: {dominant} sentiment dominates',
                'fallback_used': True
            }


# Tool registration functions for MCP server
async def get_sentiment_tool_definition():
    """Return the tool definition for MCP server registration"""
    return {
        "name": "analyze_sentiment",
        "description": "Analyze sentiment of text content using advanced Mistral LLM",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to analyze for sentiment"
                },
                "method": {
                    "type": "string",
                    "description": "Analysis method: 'llm' (default), 'detailed' for comprehensive analysis",
                    "enum": ["llm", "detailed"],
                    "default": "llm"
                }
            },
            "required": ["text"]
        }
    }


async def get_articles_sentiment_tool_definition():
    """Return the articles sentiment tool definition for MCP server registration"""
    return {
        "name": "analyze_articles_sentiment",
        "description": "Analyze sentiment of multiple news articles using Mistral LLM and provide aggregate analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "articles": {
                    "type": "array",
                    "description": "Array of news articles to analyze",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["articles"]
        }
    }


async def execute_sentiment_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the sentiment analysis tool with given arguments"""
    sentiment_tool = SentimentTool()
    return await sentiment_tool.analyze_sentiment(
        text=args.get('text', ''),
        method=args.get('method', 'llm')
    )


async def execute_articles_sentiment_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the articles sentiment analysis tool with given arguments"""
    sentiment_tool = SentimentTool()
    return await sentiment_tool.analyze_articles_sentiment(
        articles=args.get('articles', [])
    )
