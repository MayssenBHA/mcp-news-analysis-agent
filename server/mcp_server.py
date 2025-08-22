"""
FastMCP News Analysis Server - Complete Implementation
"""
from fastmcp import FastMCP
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.news_tool import NewsTool
from tools.sentiment_tool import SentimentTool
from tools.summary_tool import SummaryTool

# Create FastMCP server instance
app = FastMCP("News Analysis Server")

@app.tool()
async def fetch_news(topic: str = None, country: str = "US", language: str = "en", limit: int = 5) -> str:
    """Fetch real-time news articles by topic, country, and language."""
    try:
        news_tool = NewsTool()
        result = await news_tool.fetch_news(topic, country, language, limit)
        
        if result['success']:
            articles = []
            for article in result['articles']:
                # Extract domain from URL for better source attribution
                url = article.get('url', '')
                source_name = article.get('source', 'Unknown')
                
                # If source is unknown, try to extract from URL
                if source_name == 'Unknown' and url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            # Clean up domain (remove www, etc)
                            domain = domain.replace('www.', '')
                            source_name = domain.split('.')[0].title()
                    except:
                        pass
                
                # Format article with better source info
                article_text = f"**{article['title']}**\n{article['description']}"
                if source_name != 'Unknown':
                    article_text += f"\n📍 **Source:** {source_name}"
                article_text += f"\n🔗 **URL:** {url}"
                
                articles.append(article_text)
            
            return f"📰 Found {result['total_results']} articles:\n\n" + "\n\n---\n\n".join(articles)
        else:
            return f"❌ Failed to fetch news: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.tool()
async def analyze_sentiment(text: str, method: str = "llm") -> str:
    """Analyze sentiment of text using Mistral LLM."""
    try:
        sentiment_tool = SentimentTool()
        result = await sentiment_tool.analyze_sentiment(text, method)
        
        if result['success']:
            sentiment = result['sentiment']
            response = "🧠 **LLM Sentiment Analysis Results:**\n\n"
            
            response += f"🎭 **Classification:** {sentiment['classification'].title()}\n"
            response += f"📊 **Confidence:** {sentiment['confidence']:.2f}\n"
            response += f"� **Reasoning:** {sentiment['reasoning']}\n"
            
            if 'emotional_tone' in sentiment:
                response += f"😊 **Emotional Tone:** {sentiment['emotional_tone']}\n"
            if 'subjectivity' in sentiment:
                response += f"🎯 **Subjectivity:** {sentiment['subjectivity']}\n"
            if 'intensity' in sentiment:
                response += f"⚡ **Intensity:** {sentiment['intensity']}\n"
            if 'key_phrases' in sentiment:
                response += f"🔑 **Key Phrases:** {', '.join(sentiment['key_phrases'])}\n"
            
            return response
        else:
            return f"❌ Failed to analyze sentiment: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.tool()
async def summarize_text(text: str, max_length: int = 500, summary_type: str = "comprehensive") -> str:
    """Summarize text using Mistral AI."""
    try:
        summary_tool = SummaryTool()
        result = await summary_tool.summarize_text(text, max_length, summary_type)
        
        if result['success']:
            return f"📝 **Summary ({result['summary_type']}):**\n\n{result['summary']}\n\n📊 **Stats:** {result['original_length']} → {result['summary_length']} chars (ratio: {result['compression_ratio']})"
        else:
            return f"❌ Failed to summarize: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.tool()
async def analyze_news_sentiment(topic: str = None, country: str = "US", language: str = "en", limit: int = 3) -> str:
    """Complete workflow: fetch news and analyze sentiment of the articles."""
    try:
        from urllib.parse import urlparse
        
        # Fetch news using the tool directly
        news_tool = NewsTool()
        news_result = await news_tool.fetch_news(topic, country, language, limit)
        
        if not news_result.get('success'):
            return f"❌ Failed to fetch news: {news_result.get('error', 'Unknown error')}"
        
        articles = news_result['articles']
        if not articles:
            return f"❌ No articles found for topic: {topic or 'general news'}"
        
        # Extract text from articles for sentiment analysis
        article_texts = []
        formatted_articles = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No Title')
            description = article.get('description', 'No Description')
            url = article.get('url', '')
            source = article.get('source', 'Unknown')
            
            # Extract domain from URL if source is unknown
            if source == 'Unknown' and url:
                try:
                    domain = urlparse(url).netloc.replace('www.', '')
                    source = domain.split('.')[0].title()
                except:
                    pass
            
            # Combine title and description for sentiment analysis
            full_text = f"{title}. {description}"
            article_texts.append(full_text)
            
            # Format for display
            formatted_articles.append(f"**{i}. {title}**\n{description}\n📍 **Source:** {source}")
        
        # Analyze sentiment of combined text
        combined_text = " | ".join(article_texts)
        sentiment_tool = SentimentTool()
        sentiment_result = await sentiment_tool.analyze_sentiment(combined_text, "llm")
        
        # Build comprehensive response
        response = f"🔍 **Comprehensive News Sentiment Analysis**\n\n"
        response += f"📰 **Topic:** {topic or 'Top Headlines'}\n"
        response += f"🌍 **Region:** {country}\n"
        response += f"📊 **Articles Analyzed:** {len(articles)}\n\n"
        
        # Add news articles
        response += "📰 **Latest Articles:**\n\n"
        response += "\n\n---\n\n".join(formatted_articles)
        
        # Add LLM sentiment analysis
        response += "\n\n🎭 **LLM Sentiment Analysis Results:**\n\n"
        
        if sentiment_result.get('success'):
            sentiment = sentiment_result['sentiment']
            response += f"🎭 **Overall Classification:** {sentiment['classification'].title()}\n"
            response += f"� **Confidence:** {sentiment['confidence']:.2f}\n"
            response += f"💭 **AI Reasoning:** {sentiment['reasoning']}\n"
            
            # Add interpretation
            classification = sentiment['classification'].lower()
            if 'positive' in classification:
                response += f"😊 **Interpretation:** News coverage appears generally optimistic or favorable about {topic or 'the topic'}."
            elif 'negative' in classification:
                response += f"😔 **Interpretation:** News coverage appears generally critical or concerning about {topic or 'the topic'}."
            else:
                response += f"😐 **Interpretation:** News coverage appears balanced or neutral about {topic or 'the topic'}."
        else:
            response += f"❌ Sentiment analysis failed: {sentiment_result.get('error', 'Unknown error')}"
        
        return response
        
    except Exception as e:
        return f"❌ Error in combined analysis: {str(e)}"

if __name__ == "__main__":
    # Use the proper FastMCP stdio async runner
    asyncio.run(app.run_stdio_async(show_banner=False))
