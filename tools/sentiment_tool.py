"""
Sentiment Analysis Tool - Analyzes sentiment of text or news articles
"""

from typing import Dict, Any, List
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import asyncio


class SentimentTool:
    """Tool for analyzing sentiment of text content"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    async def analyze_sentiment(self, text: str, method: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze sentiment of given text
        
        Args:
            text: Text to analyze
            method: Analysis method ('textblob', 'vader', 'comprehensive')
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'No text provided for sentiment analysis'
            }
        
        try:
            result = {'success': True, 'text_length': len(text)}
            
            if method in ['textblob', 'comprehensive']:
                # TextBlob analysis
                blob = TextBlob(text)
                textblob_sentiment = {
                    'polarity': round(blob.sentiment.polarity, 3),  # -1 to 1
                    'subjectivity': round(blob.sentiment.subjectivity, 3),  # 0 to 1
                    'classification': self._classify_textblob_sentiment(blob.sentiment.polarity)
                }
                result['textblob'] = textblob_sentiment
            
            if method in ['vader', 'comprehensive']:
                # VADER analysis
                vader_scores = self.vader_analyzer.polarity_scores(text)
                vader_sentiment = {
                    'compound': round(vader_scores['compound'], 3),
                    'positive': round(vader_scores['pos'], 3),
                    'neutral': round(vader_scores['neu'], 3),
                    'negative': round(vader_scores['neg'], 3),
                    'classification': self._classify_vader_sentiment(vader_scores['compound'])
                }
                result['vader'] = vader_sentiment
            
            # Add overall classification if comprehensive analysis
            if method == 'comprehensive':
                result['overall_sentiment'] = self._get_overall_sentiment(result)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Sentiment analysis failed: {str(e)}'
            }
    
    async def analyze_articles_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple news articles
        
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
                
                sentiment_result = await self.analyze_sentiment(text_to_analyze, method='comprehensive')
                
                if sentiment_result['success']:
                    article_sentiment = {
                        'article_index': i,
                        'title': article.get('title', 'No title')[:100],
                        'sentiment': sentiment_result
                    }
                    article_sentiments.append(article_sentiment)
            
            # Calculate aggregate sentiment
            aggregate_sentiment = self._calculate_aggregate_sentiment(article_sentiments)
            
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
    
    def _classify_textblob_sentiment(self, polarity: float) -> str:
        """Classify TextBlob polarity score into sentiment category"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_vader_sentiment(self, compound: float) -> str:
        """Classify VADER compound score into sentiment category"""
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_overall_sentiment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall sentiment from TextBlob and VADER results"""
        textblob_class = analysis_result.get('textblob', {}).get('classification', 'neutral')
        vader_class = analysis_result.get('vader', {}).get('classification', 'neutral')
        
        # Simple majority vote
        if textblob_class == vader_class:
            classification = textblob_class
            confidence = 'high'
        elif (textblob_class == 'neutral' and vader_class != 'neutral') or \
             (textblob_class != 'neutral' and vader_class == 'neutral'):
            classification = textblob_class if textblob_class != 'neutral' else vader_class
            confidence = 'medium'
        else:
            classification = 'mixed'
            confidence = 'low'
        
        return {
            'classification': classification,
            'confidence': confidence,
            'textblob_agrees': textblob_class,
            'vader_agrees': vader_class
        }
    
    def _calculate_aggregate_sentiment(self, article_sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate sentiment across multiple articles"""
        if not article_sentiments:
            return {'classification': 'neutral', 'confidence': 'low'}
        
        classifications = []
        textblob_polarities = []
        vader_compounds = []
        
        for article_sentiment in article_sentiments:
            sentiment_data = article_sentiment['sentiment']
            
            if 'overall_sentiment' in sentiment_data:
                classifications.append(sentiment_data['overall_sentiment']['classification'])
            
            if 'textblob' in sentiment_data:
                textblob_polarities.append(sentiment_data['textblob']['polarity'])
            
            if 'vader' in sentiment_data:
                vader_compounds.append(sentiment_data['vader']['compound'])
        
        # Calculate averages
        avg_textblob = sum(textblob_polarities) / len(textblob_polarities) if textblob_polarities else 0
        avg_vader = sum(vader_compounds) / len(vader_compounds) if vader_compounds else 0
        
        # Count classifications
        positive_count = classifications.count('positive')
        negative_count = classifications.count('negative')
        neutral_count = classifications.count('neutral')
        mixed_count = classifications.count('mixed')
        
        total_articles = len(classifications)
        
        return {
            'average_textblob_polarity': round(avg_textblob, 3),
            'average_vader_compound': round(avg_vader, 3),
            'sentiment_distribution': {
                'positive': f"{positive_count}/{total_articles} ({positive_count/total_articles*100:.1f}%)",
                'negative': f"{negative_count}/{total_articles} ({negative_count/total_articles*100:.1f}%)",
                'neutral': f"{neutral_count}/{total_articles} ({neutral_count/total_articles*100:.1f}%)",
                'mixed': f"{mixed_count}/{total_articles} ({mixed_count/total_articles*100:.1f}%)"
            },
            'dominant_sentiment': max(['positive', 'negative', 'neutral', 'mixed'], 
                                   key=lambda x: classifications.count(x))
        }


# Tool registration functions for MCP server
async def get_sentiment_tool_definition():
    """Return the tool definition for MCP server registration"""
    return {
        "name": "analyze_sentiment",
        "description": "Analyze sentiment of text content using TextBlob and VADER sentiment analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to analyze for sentiment"
                },
                "method": {
                    "type": "string",
                    "description": "Analysis method: 'textblob', 'vader', or 'comprehensive'",
                    "enum": ["textblob", "vader", "comprehensive"],
                    "default": "comprehensive"
                }
            },
            "required": ["text"]
        }
    }


async def get_articles_sentiment_tool_definition():
    """Return the articles sentiment tool definition for MCP server registration"""
    return {
        "name": "analyze_articles_sentiment",
        "description": "Analyze sentiment of multiple news articles and provide aggregate analysis",
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
        method=args.get('method', 'comprehensive')
    )


async def execute_articles_sentiment_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the articles sentiment analysis tool with given arguments"""
    sentiment_tool = SentimentTool()
    return await sentiment_tool.analyze_articles_sentiment(
        articles=args.get('articles', [])
    )
