"""
MCP Client/Agent with LangChain Integration
Uses LLM-powered reasoning instead of rule-based intent detection
"""

import asyncio
import json
import logging
import sys
import os
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from contextlib import AsyncExitStack

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

# MCP client imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    print("✅ MCP imports successful")
except ImportError:
    print("MCP client packages not installed. Please install with: pip install mcp")
    exit(1)

from config.settings import config, validate_config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryIntent(BaseModel):
    """Structured output for query intent classification"""
    intent: str = Field(description="Primary intent: fetch_news, sentiment_analysis, summarize, combined_analysis")
    confidence: float = Field(description="Confidence score between 0 and 1")
    topic: Optional[str] = Field(description="News topic if specified")
    text: Optional[str] = Field(description="Specific text to analyze (if provided in quotes)")
    country: Optional[str] = Field(description="Country code if specified (US, UK, FR, etc.)")
    language: Optional[str] = Field(description="Language code if specified (en, fr, es, etc.)")
    limit: Optional[int] = Field(description="Number of articles requested")
    reasoning: str = Field(description="Brief explanation of the classification")
    follow_up_suggestions: List[str] = Field(description="Suggested follow-up queries")


class SimpleConversationMemory:
    """Simple conversation memory implementation"""
    
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages
    
    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
        self._trim_messages()
    
    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))
        self._trim_messages()
    
    def _trim_messages(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_chat_history_string(self) -> str:
        return "\n".join([f"{msg.type}: {msg.content[:200]}..." if len(msg.content) > 200 else f"{msg.type}: {msg.content}" 
                         for msg in self.messages[-6:]])


class ConversationContext(BaseModel):
    """Context for conversation memory"""
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    recent_topics: List[str] = Field(default_factory=list)
    last_results: Optional[Dict[str, Any]] = Field(default=None)


class EnhancedMCPAgent:
    """Enhanced MCP agent with LangChain-powered reasoning"""
    
    def __init__(self):
        self.mcp_session = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize LangChain components
        if config.mistral_api_key:
            self.llm = ChatMistralAI(
                api_key=config.mistral_api_key,
                model="mistral-small-latest",
                temperature=0.1
            )
        else:
            logger.warning("No Mistral API key found. LLM features will be disabled.")
            self.llm = None
        
        # Conversation memory
        self.memory = SimpleConversationMemory(max_messages=20)
        
        # Conversation context
        self.context = ConversationContext()
        
        # Initialize chains
        self._setup_chains()
        
        # Available MCP tools (will be populated after connection)
        self.mcp_tools = {}
    
    def _setup_chains(self):
        """Setup LangChain chains for different tasks"""
        
        if not self.llm:
            logger.warning("Cannot setup LangChain chains without LLM")
            return
        
        # Intent classification chain
        intent_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are an expert news analysis assistant. Analyze user queries and classify their intent with high accuracy.

Available intents:
1. fetch_news - User wants to get news articles
2. sentiment_analysis - User wants sentiment analysis of news or specific text
3. summarize - User wants summaries of news or text
4. combined_analysis - User wants comprehensive news analysis with sentiment

CRITICAL DETECTION RULES:
- If the query contains text in quotes (single or double), extract that text for direct analysis
- "Analyze sentiment: 'text here'" → extract 'text here' for direct sentiment analysis
- If no quotes, user wants sentiment analysis of news on a topic

Required JSON format:
{{
  "intent": "fetch_news|sentiment_analysis|summarize|combined_analysis",
  "confidence": 0.95,
  "topic": "specific subject or null",
  "text": "exact quoted text for direct analysis or null",
  "country": "US|UK|FR|DE|CA|AU|JP|CN|null",
  "language": "en|fr|es|de|etc",
  "limit": 5,
  "reasoning": "Brief explanation",
  "follow_up_suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}

Examples:
Input: "Analyze sentiment: 'This product is amazing but expensive'"
Output: {{"intent": "sentiment_analysis", "confidence": 0.98, "topic": null, "text": "This product is amazing but expensive", "country": null, "language": "en", "limit": 5, "reasoning": "User provides specific text for direct sentiment analysis", "follow_up_suggestions": ["Get product reviews news", "Analyze market trends", "Summarize product feedback"]}}

Input: "How do people feel about climate change?"
Output: {{"intent": "sentiment_analysis", "confidence": 0.95, "topic": "climate change", "text": null, "country": null, "language": "en", "limit": 5, "reasoning": "User asks about public sentiment on climate change from news", "follow_up_suggestions": ["Get latest climate news", "Summarize climate reports", "Analyze climate policy news"]}}

Current conversation context: {context}
Chat history: {chat_history}
            """),
            HumanMessagePromptTemplate.from_template("User query: {query}")
        ])
        
        self.intent_chain = intent_prompt | self.llm | JsonOutputParser(pydantic_object=QueryIntent)
        
        # Query enhancement chain
        enhancement_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a query enhancement specialist. Improve user queries for better news search results.

Guidelines:
- Expand abbreviations and acronyms
- Add relevant keywords
- Fix typos and grammar
- Maintain original intent
- Make queries more specific but not too narrow

Context: {context}
Chat history: {chat_history}
            """),
            HumanMessagePromptTemplate.from_template("""
Original query: {query}
Detected intent: {intent}
Topic: {topic}

Provide an enhanced version of this query for better news search results.
Return only the enhanced query, nothing else.
            """)
        ])
        
        self.enhancement_chain = enhancement_prompt | self.llm | StrOutputParser()
        
        # Result analysis chain
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a news analysis expert. Analyze news results and provide insights.

Your task:
- Identify key themes and trends
- Highlight important information
- Suggest related topics
- Provide context and background

Context: {context}
Chat history: {chat_history}
            """),
            HumanMessagePromptTemplate.from_template("""
Query: {query}
Results: {results}

Provide a brief analysis of these news results, highlighting key insights and trends.
Keep it concise but informative.
            """)
        ])
        
        self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()

    async def start_session(self):
        """Initialize MCP connection and setup tools"""
        try:
            # Start MCP server
            server_params = StdioServerParameters(
                command="python",
                args=["server/mcp_server.py"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            logger.info(f"Starting server: python server/mcp_server.py")
            
            # Use AsyncExitStack for proper resource management
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            read, write = stdio_transport
            logger.info("✅ STDIO transport established")
            
            # Create client session with the streams
            self.mcp_session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            logger.info("✅ Client session created")
            
            # Initialize the session
            logger.info("🔄 Initializing MCP session...")
            
            try:
                await asyncio.wait_for(self.mcp_session.initialize(), timeout=10.0)
                logger.info("✅ MCP session initialized successfully!")
            except asyncio.TimeoutError:
                logger.error("❌ MCP session initialization timed out")
                return False
            except Exception as init_error:
                logger.error(f"❌ MCP session initialization failed: {init_error}")
                return False
            
            # List available tools and create LangChain wrappers
            try:
                logger.info("🔄 Retrieving available tools...")
                tools_response = await asyncio.wait_for(self.mcp_session.list_tools(), timeout=5.0)
                available_tools = [tool.name for tool in tools_response.tools]
                logger.info(f"🛠️  Available tools: {available_tools}")
                
                if not available_tools:
                    logger.warning("⚠️  No tools found - server may not be responding correctly")
                    return False
                
                # Create LangChain tool wrappers
                await self._setup_mcp_tools(tools_response.tools)
                
            except Exception as e:
                logger.warning(f"⚠️  Could not list tools: {e}")
                return False
            
            logger.info("🎉 Enhanced MCP connection established successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start MCP session: {e}")
            return False

    async def _setup_mcp_tools(self, mcp_tools):
        """Create LangChain Tool wrappers for MCP tools"""
        
        async def call_mcp_tool(tool_name: str, **kwargs) -> str:
            """Wrapper function to call MCP tools"""
            try:
                result = await self.mcp_session.call_tool(tool_name, kwargs)
                if hasattr(result.content[0], 'text'):
                    return result.content[0].text
                else:
                    return str(result.content[0])
            except Exception as e:
                return f"Error calling {tool_name}: {str(e)}"
        
        # Create LangChain tools for each MCP tool
        for tool in mcp_tools:
            name = tool.name
            description = tool.description or f"MCP tool: {name}"
            
            if name == "fetch_news":
                self.mcp_tools["fetch_news"] = Tool(
                    name="fetch_news",
                    description="Fetch latest news articles by topic, country, and language",
                    func=lambda **kwargs: call_mcp_tool("fetch_news", **kwargs)
                )
            elif name == "analyze_sentiment":
                self.mcp_tools["analyze_sentiment"] = Tool(
                    name="analyze_sentiment",
                    description="Analyze sentiment of text using Mistral LLM",
                    func=lambda **kwargs: call_mcp_tool("analyze_sentiment", **kwargs)
                )
            elif name == "summarize_text":
                self.mcp_tools["summarize_text"] = Tool(
                    name="summarize_text",
                    description="Summarize text using Mistral AI",
                    func=lambda **kwargs: call_mcp_tool("summarize_text", **kwargs)
                )
            elif name == "analyze_news_sentiment":
                self.mcp_tools["analyze_news_sentiment"] = Tool(
                    name="analyze_news_sentiment",
                    description="Complete workflow: fetch news and analyze sentiment",
                    func=lambda **kwargs: call_mcp_tool("analyze_news_sentiment", **kwargs)
                )

    async def close_session(self):
        """Clean up MCP session and resources"""
        try:
            logger.info("🔄 Closing enhanced MCP session...")
            
            # Clean up context manager stack
            if hasattr(self.exit_stack, 'close'):
                await self.exit_stack.close()
            else:
                # Handle potential cleanup errors gracefully
                try:
                    await self.exit_stack.aclose()
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup warning (safe to ignore): {cleanup_error}")
            
            logger.info("✅ Enhanced MCP client cleanup complete")
            
        except Exception as e:
            logger.warning(f"Warning during cleanup (safe to ignore): {e}")

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process natural language query using LangChain-powered reasoning
        
        Args:
            query: Natural language query from user
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Handle special commands first
            query_lower = query.lower().strip()
            
            # Handle help commands
            if (query_lower in ['help', '?', 'examples', 'usage', 'commands'] or 
                'help' in query_lower or 'example' in query_lower):
                return {
                    'success': True,
                    'response': self._get_help_response(),
                    'intent': 'help',
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Handle comments/test queries
            if query.strip().startswith('#'):
                return {
                    'success': True,
                    'response': "This appears to be a comment or test query. Please provide a real news query, or type 'help' for examples.",
                    'intent': 'comment',
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Validate query
            if not query.strip() or len(query.strip()) < 2:
                return {
                    'success': False,
                    'error': 'Query too short. Please provide a more detailed request.'
                }
            
            logger.info(f"Processing query: {query}")
            
            # Use LangChain to analyze query intent
            intent_result = await self._analyze_intent_with_langchain(query)
            
            if not intent_result:
                return {
                    'success': False,
                    'error': 'Failed to understand query intent'
                }
            
            # Execute workflow based on intent
            workflow_result = await self._execute_enhanced_workflow(query, intent_result)
            
            # Store results in context for future reference
            self.context.last_results = workflow_result
            if intent_result.topic:
                if intent_result.topic not in self.context.recent_topics:
                    self.context.recent_topics.append(intent_result.topic)
                    # Keep only last 5 topics
                    self.context.recent_topics = self.context.recent_topics[-5:]
            
            # Add to conversation memory
            self.memory.add_user_message(query)
            
            return {
                'success': True,
                'query': query,
                'intent': intent_result.intent,
                'confidence': intent_result.confidence,
                'reasoning': intent_result.reasoning,
                'follow_up_suggestions': intent_result.follow_up_suggestions,
                'response': workflow_result['response'],
                'metadata': {
                    'topic': intent_result.topic,
                    'country': intent_result.country,
                    'language': intent_result.language,
                    'limit': intent_result.limit
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f"Failed to process query: {str(e)}",
                'query': query,
                'timestamp': datetime.now().isoformat()
            }

    async def _analyze_intent_with_langchain(self, query: str) -> Optional[QueryIntent]:
        """Use LangChain to analyze query intent"""
        try:
            if not self.llm:
                # Fallback to simple rule-based if no LLM
                return self._fallback_intent_analysis(query)
            
            # Prepare context
            context_str = json.dumps({
                "recent_topics": self.context.recent_topics,
                "user_preferences": self.context.user_preferences,
                "last_query_results": bool(self.context.last_results)
            })
            
            # Get chat history
            chat_history_str = self.memory.get_chat_history_string()
            
            # Invoke the intent classification chain
            result = await self.intent_chain.ainvoke({
                "query": query,
                "context": context_str,
                "chat_history": chat_history_str
            })
            
            return QueryIntent(**result)
            
        except Exception as e:
            logger.warning(f"LangChain intent analysis failed: {e}")
            return self._fallback_intent_analysis(query)

    def _fallback_intent_analysis(self, query: str) -> QueryIntent:
        """Fallback intent analysis when LLM is not available"""
        query_lower = query.lower()
        
        # Check for specific text analysis (text in quotes)
        import re
        text_match = re.search(r'["\']([^"\']+)["\']', query)
        
        # Simple rule-based classification
        if any(word in query_lower for word in ['feel', 'sentiment', 'opinion', 'think']):
            intent = 'sentiment_analysis'
            confidence = 0.7
            # If there's quoted text, use it directly
            if text_match:
                return QueryIntent(
                    intent=intent,
                    confidence=confidence,
                    text=text_match.group(1),  # Extract the quoted text
                    reasoning="Detected quoted text for direct sentiment analysis"
                )
        elif any(word in query_lower for word in ['summarize', 'summary', 'brief', 'tldr']):
            intent = 'summarize'
            confidence = 0.7
        elif any(word in query_lower for word in ['analyze', 'analysis', 'comprehensive']):
            intent = 'combined_analysis'
            confidence = 0.8
        else:
            intent = 'fetch_news'
            confidence = 0.6
        
        # Extract topic
        topic = None
        for keyword in ['technology', 'tech', 'ai', 'artificial intelligence', 'climate', 'politics', 'business', 'sports']:
            if keyword in query_lower:
                topic = keyword
                break
        
        # Extract country if mentioned
        country = None
        if 'us' in query_lower or 'usa' in query_lower or 'america' in query_lower:
            country = 'us'
        elif 'uk' in query_lower or 'britain' in query_lower:
            country = 'gb'
        elif 'canada' in query_lower:
            country = 'ca'

        return QueryIntent(
            intent=intent,
            confidence=confidence,
            topic=topic or "general",
            country=country,
            language="en",
            limit=5,
            reasoning=f"Rule-based classification: detected '{intent}' intent",
            follow_up_suggestions=["Try asking for help to see more examples"]
        )

    async def _execute_enhanced_workflow(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """Execute workflow with enhanced capabilities"""
        try:
            # Prepare parameters
            params = {
                'topic': intent.topic,
                'country': intent.country or 'US',
                'language': intent.language or 'en', 
                'limit': intent.limit or 5
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            if intent.intent == 'fetch_news':
                result = await self._call_mcp_tool('fetch_news', params)
                
            elif intent.intent == 'sentiment_analysis':
                if intent.text:
                    # Direct text analysis (user provided text in quotes)
                    sentiment_result = await self._call_mcp_tool('analyze_sentiment', {'text': intent.text})
                    enhanced_sentiment = self._add_sentiment_summary(sentiment_result.get('result', 'Analysis failed'))
                    result = {
                        'success': sentiment_result.get('success', False),
                        'result': enhanced_sentiment
                    }
                elif 'text' in params:
                    # Fallback: text provided in params
                    sentiment_result = await self._call_mcp_tool('analyze_sentiment', {'text': params['text']})
                    enhanced_sentiment = self._add_sentiment_summary(sentiment_result.get('result', 'Analysis failed'))
                    result = {
                        'success': sentiment_result.get('success', False),
                        'result': enhanced_sentiment
                    }
                else:
                    # Fetch news first, then analyze sentiment
                    news_result = await self._call_mcp_tool('fetch_news', params)
                    if news_result.get('success'):
                        # Extract text for analysis
                        news_text = self._extract_text_from_results(news_result['result'])
                        sentiment_result = await self._call_mcp_tool('analyze_sentiment', {'text': news_text})
                        
                        # Add enhanced sentiment summary
                        enhanced_sentiment = self._add_sentiment_summary(sentiment_result.get('result', 'Analysis failed'))
                        
                        # Use clean text extraction for news content
                        clean_news = self._extract_clean_text_content(news_result['result'])
                        
                        result = {
                            'success': True,
                            'result': clean_news + "\n\n🎭 **Sentiment Analysis:**\n" + enhanced_sentiment
                        }
                    else:
                        result = news_result
                        
            elif intent.intent == 'summarize':
                # Similar logic for summarization
                news_result = await self._call_mcp_tool('fetch_news', params)
                if news_result.get('success'):
                    news_text = self._extract_text_from_results(news_result['result'])
                    summary_result = await self._call_mcp_tool('summarize_text', {'text': news_text})
                    
                    # Use clean text extraction for news content
                    clean_news = self._extract_clean_text_content(news_result['result'])
                    clean_summary = self._extract_clean_text_content(summary_result.get('result', 'Summary failed'))
                    
                    result = {
                        'success': True,
                        'result': clean_news + "\n\n📝 **Summary:**\n" + clean_summary
                    }
                else:
                    result = news_result
                    
            elif intent.intent == 'combined_analysis':
                result = await self._call_mcp_tool('analyze_news_sentiment', params)
                
            else:
                result = {
                    'success': False,
                    'result': f'Unknown intent: {intent.intent}'
                }
            
            # Format response
            formatted_response = await self._format_enhanced_response(result, intent, query)
            
            return {
                'success': result.get('success', False),
                'response': formatted_response
            }
            
        except Exception as e:
            logger.error(f"Enhanced workflow execution failed: {e}")
            return {
                'success': False,
                'response': f"Workflow execution failed: {str(e)}"
            }

    async def _call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server tool with given parameters"""
        try:
            if not self.mcp_session:
                return {
                    'success': False,
                    'error': 'MCP session not initialized'
                }
            
            # Call the tool
            result = await self.mcp_session.call_tool(tool_name, parameters)
            
            return {
                'success': True,
                'tool': tool_name,
                'result': result.content,
                'parameters_used': parameters
            }
            
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return {
                'success': False,
                'error': f'Tool call failed: {str(e)}'
            }

    def _extract_text_from_results(self, results) -> str:
        """Extract text content from MCP results for further analysis"""
        if isinstance(results, list) and len(results) > 0:
            if hasattr(results[0], 'text'):
                return results[0].text[:2000]  # Limit length
            else:
                return str(results[0])[:2000]
        else:
            return str(results)[:2000]
    
    def _extract_clean_text_content(self, content) -> str:
        """Extract and clean text content from MCP results, formatting as structured output"""
        # Handle TextContent objects from MCP results
        if isinstance(content, list):
            if len(content) > 0 and hasattr(content[0], 'text'):
                raw_text = content[0].text
            else:
                raw_text = str(content[0]) if content else "No content"
        elif not isinstance(content, str):
            if hasattr(content, 'text'):
                raw_text = content.text
            else:
                raw_text = str(content)
        else:
            raw_text = content
        
        # Parse and structure the content by articles
        return self._structure_article_content(raw_text)
    
    def _structure_article_content(self, raw_text: str) -> str:
        """Convert raw article text into structured bullet points by article"""
        if not raw_text or raw_text == "No content":
            return raw_text
        
        # Check if this contains multiple articles (news format)
        if "📰 Found" in raw_text and "articles:" in raw_text:
            return self._format_news_articles_structured(raw_text)
        
        # For sentiment/summary results, return as-is but cleaned
        return raw_text.replace("\\n", "\n").replace("\\xa0", " ")
    
    def _format_news_articles_structured(self, content: str) -> str:
        """Format news articles in a clean, structured bullet point format"""
        lines = content.split('\n')
        articles = []
        current_article = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('📰 Found'):
                continue
                
            # New article starts with **title**
            if line.startswith('**') and line.endswith('**'):
                # Save previous article if exists
                if current_article:
                    articles.append(current_article)
                # Start new article
                title = line.strip('**')
                current_article = {'title': title, 'description': '', 'source': '', 'url': ''}
            
            # Article description (paragraph after title)
            elif current_article and not line.startswith('📍') and not line.startswith('🔗') and not line.startswith('---'):
                if 'description' in current_article and not current_article['description']:
                    current_article['description'] = line
            
            # Source information
            elif line.startswith('📍 **Source:**'):
                if current_article:
                    current_article['source'] = line.replace('📍 **Source:**', '').strip()
            
            # URL information
            elif line.startswith('🔗 **URL:**'):
                if current_article:
                    current_article['url'] = line.replace('🔗 **URL:**', '').strip()
        
        # Don't forget the last article
        if current_article:
            articles.append(current_article)
        
        # Format as structured output
        if not articles:
            return content  # fallback to original if parsing failed
        
        formatted_output = f"📰 **Found {len(articles)} articles:**\n\n"
        
        for i, article in enumerate(articles, 1):
            formatted_output += f"### 📄 **Article {i}**\n"
            formatted_output += f"• **Title:** {article.get('title', 'N/A')}\n"
            if article.get('description'):
                formatted_output += f"• **Summary:** {article.get('description', 'N/A')}\n"
            formatted_output += f"• **Source:** {article.get('source', 'N/A')}\n"
            formatted_output += f"• **URL:** {article.get('url', 'N/A')}\n\n"
        
        return formatted_output

    def _add_sentiment_summary(self, content) -> str:
        """Add a summary table for sentiment analysis results"""
        # Extract clean text content
        clean_content = self._extract_clean_text_content(content)
        
        # Extract sentiment information if available
        sentiment_summary = """
### 📊 Sentiment Summary

| Method | Result | Score |
|--------|---------|--------|
"""
        
        lines = clean_content.split('\n')
        for line in lines:
            if 'TextBlob:' in line:
                method = "TextBlob"
                result_part = line.split('TextBlob:', 1)[1].strip()
                if 'negative' in result_part.lower():
                    emoji = "😔 Negative"
                elif 'positive' in result_part.lower():
                    emoji = "😊 Positive"  
                else:
                    emoji = "😐 Neutral"
                
                # Extract polarity if present
                polarity = "N/A"
                if 'polarity:' in result_part:
                    polarity = result_part.split('polarity:')[1].split(')')[0].strip()
                
                sentiment_summary += f"| {method} | {emoji} | {polarity} |\n"
                
            elif 'VADER:' in line:
                method = "VADER"
                result_part = line.split('VADER:', 1)[1].strip()
                if 'negative' in result_part.lower():
                    emoji = "😔 Negative"
                elif 'positive' in result_part.lower():
                    emoji = "😊 Positive"
                else:
                    emoji = "😐 Neutral"
                
                # Extract compound score if present  
                compound = "N/A"
                if 'compound:' in result_part:
                    compound = result_part.split('compound:')[1].split(')')[0].strip()
                
                sentiment_summary += f"| {method} | {emoji} | {compound} |\n"
        
        return clean_content + "\n" + sentiment_summary

    def _format_news_articles(self, content: str, intent: QueryIntent) -> str:
        """Format news articles with rich markdown formatting"""
        lines = content.split('\n')
        formatted_lines = []
        article_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**') and len(line) > 4 and not line.startswith('**Source:**') and not line.startswith('**URL:**'):
                # Article title
                article_count += 1
                formatted_lines.append(f"\n### 📰 Article #{article_count}: {line}")
            elif line.startswith('📍 **Source:**') or line.startswith('**📍 Source:**'):
                # Source with emoji
                source = line.replace('📍 **Source:**', '').replace('**📍 Source:**', '').strip()
                formatted_lines.append(f"**📍 Source:** {source}")
            elif line.startswith('🔗 **URL:**') or line.startswith('**🔗 Link:**'):
                # URL formatting
                url = line.replace('🔗 **URL:**', '').replace('**🔗 Link:**', '').strip()
                formatted_lines.append(f"**🔗 Link:** [{url}]({url})")
            elif line == '---':
                # Add spacing between articles
                formatted_lines.append('\n' + '─' * 50 + '\n')
            elif line and not line.startswith('📰 Found') and not line.startswith('�') and not line.startswith('🔗'):
                # Article description/content (but not source/url lines)
                if not line.startswith('**Source:**') and not line.startswith('**URL:**'):
                    formatted_lines.append(f"**📄 Summary:** {line}")
            elif line:
                # Keep other content as-is
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def _format_sentiment_results(self, content, include_explanation: bool = True) -> str:
        """Format sentiment analysis results with explanations"""
        # Extract text content from MCP result format
        actual_content = self._extract_clean_text_content(content)
        
        formatted = actual_content
        
        if include_explanation:
            explanation = """
**🔬 How Sentiment Analysis Works:**
• **TextBlob:** Uses machine learning to analyze text polarity (-1.0 to +1.0)
• **VADER:** Valence Aware Dictionary for sentiment intensity analysis
• **Overall:** Combines both methods with confidence weighting
• **Scale:** Negative (-1.0) ← Neutral (0.0) → Positive (+1.0)
"""
            formatted += explanation
            
        # Add emoji indicators for sentiment scores
        lines = formatted.split('\n')
        formatted_lines = []
        
        for line in lines:
            if 'TextBlob:' in line:
                if 'negative' in line.lower():
                    formatted_lines.append(f"📊 😔 **TextBlob:** {line.split('TextBlob:', 1)[1].strip()}")
                elif 'positive' in line.lower():
                    formatted_lines.append(f"📊 😊 **TextBlob:** {line.split('TextBlob:', 1)[1].strip()}")
                else:
                    formatted_lines.append(f"📊 😐 **TextBlob:** {line.split('TextBlob:', 1)[1].strip()}")
            elif 'VADER:' in line:
                if 'negative' in line.lower():
                    formatted_lines.append(f"📈 😔 **VADER:** {line.split('VADER:', 1)[1].strip()}")
                elif 'positive' in line.lower():
                    formatted_lines.append(f"📈 😊 **VADER:** {line.split('VADER:', 1)[1].strip()}")
                else:
                    formatted_lines.append(f"📈 😐 **VADER:** {line.split('VADER:', 1)[1].strip()}")
            elif 'Overall:' in line:
                if 'negative' in line.lower():
                    formatted_lines.append(f"🎯 😔 **Overall Sentiment:** {line.split('Overall:', 1)[1].strip()}")
                elif 'positive' in line.lower():
                    formatted_lines.append(f"🎯 😊 **Overall Sentiment:** {line.split('Overall:', 1)[1].strip()}")
                else:
                    formatted_lines.append(f"🎯 😐 **Overall Sentiment:** {line.split('Overall:', 1)[1].strip()}")
            else:
                formatted_lines.append(line)
                
        return '\n'.join(formatted_lines)

    def _format_summary_results(self, content, include_explanation: bool = True) -> str:
        """Format summary results with explanations"""
        # Extract text content from MCP result format
        actual_content = self._extract_clean_text_content(content)
            
        formatted = actual_content
        
        if include_explanation:
            explanation = """
**🔬 How Text Summarization Works:**
• **Extractive Method:** Identifies and extracts the most important sentences
• **Frequency Analysis:** Uses word frequency and sentence ranking algorithms  
• **Key Phrases:** Preserves essential information while reducing length
• **Context Preservation:** Maintains logical flow and readability
"""
            formatted += explanation
            
        return formatted

    async def _format_enhanced_response(self, result: Dict[str, Any], intent: QueryIntent, query: str) -> str:
        """Format response with enhanced capabilities"""
        if not result.get('success'):
            return f"I apologize, but I encountered an error: {result.get('error', 'Unknown error')}"
        
        # Extract raw content from MCP result
        raw_result = result.get('result', 'No results')
        
        # Use clean text extraction for all content
        content = self._extract_clean_text_content(raw_result)
        
        # Add LLM-powered analysis if available
        enhanced_content = content
        if self.llm and intent.intent in ['fetch_news', 'combined_analysis']:
            try:
                # For analysis, use the raw text content
                analysis_input = content if isinstance(content, str) else str(content)
                analysis = await self.analysis_chain.ainvoke({
                    'query': query,
                    'results': analysis_input,
                    'context': json.dumps(self.context.model_dump()),
                    'chat_history': ''
                })
                enhanced_content += f"\n\n💡 **AI Analysis:**\n{analysis}"
            except Exception as e:
                logger.warning(f"Analysis generation failed: {e}")
        
        # Format based on intent with rich formatting
        if intent.intent == 'fetch_news':
            header = f"## 📰 Latest News"
            if intent.topic:
                header += f" about **{intent.topic.title()}**"
            if intent.country:
                header += f" from **{intent.country.upper()}**"
            header += "\n\n"
            
            # Content is already formatted by _extract_clean_text_content
            return header + enhanced_content
            
        elif intent.intent == 'sentiment_analysis':
            header = "## 🎭 Sentiment Analysis Results\n\n"
            # Apply rich sentiment formatting with explanations
            formatted_sentiment = self._format_sentiment_results(enhanced_content, include_explanation=True)
            return header + formatted_sentiment
            
        elif intent.intent == 'summarize':
            header = "## 📝 Text Summary\n\n"
            # Apply rich summary formatting with explanations
            formatted_summary = self._format_summary_results(enhanced_content, include_explanation=True)
            return header + formatted_summary
            
        elif intent.intent == 'combined_analysis':
            header = "## 🔍 Comprehensive News & Sentiment Analysis\n\n"
            
            # Check if this contains both news and sentiment
            if '🎭 **Sentiment Analysis:**' in enhanced_content:
                parts = enhanced_content.split('🎭 **Sentiment Analysis:**')
                news_part = parts[0].strip()
                sentiment_part = parts[1].strip() if len(parts) > 1 else ""
                
                formatted_news = self._format_news_articles(news_part, intent)
                formatted_sentiment = self._format_sentiment_results(sentiment_part, include_explanation=True)
                
                return header + formatted_news + "\n\n### 🎭 Sentiment Analysis\n\n" + formatted_sentiment
            else:
                return header + enhanced_content
        
        return enhanced_content

    def _get_help_response(self) -> str:
        """Generate comprehensive help response"""
        return """🤖 **Enhanced MCP News Analysis Agent - Help Guide**

🧠 **AI-Powered Features:**
• 🎯 Intelligent query understanding with LLM reasoning
• 💬 Conversation memory and context awareness
• 🔄 Smart tool selection and orchestration  
• 📊 Advanced result analysis and insights

📰 **News Analysis Capabilities:**
• 📰 Fetch latest news by topic or location
• 🎭 Analyze sentiment of news or custom text  
• 📝 Summarize news articles or text
• 🔍 Combined analysis (news + sentiment + insights)

💬 **Natural Language Examples:**
• "What's the latest in AI?" → Intelligent topic detection
• "How are people reacting to climate policies?" → Sentiment analysis
• "Give me a rundown of today's tech news" → Summary generation
• "I want to understand the mood around cryptocurrency" → Combined analysis

🎯 **Smart Features:**
• **Context Awareness**: Remembers previous queries and preferences  
• **Follow-up Support**: "Tell me more about that" or "What about sports instead?"
• **Auto-suggestions**: Get related query suggestions after each response
• **Enhanced Understanding**: Handles typos, synonyms, and complex requests

⚡ **Quick Commands:**
• "tech" → Technology news
• "ai news" → Artificial intelligence coverage
• "climate sentiment" → Climate change opinion analysis
• "business summary" → Business news overview
• "help" or "?" → Show this guide

🌍 **Global Coverage:**
• Specify regions: "news from Europe", "US politics", "Asian markets"
• Language options: "French news", "Spanish updates"  
• Custom limits: "top 10 articles", "latest 3 stories"

💡 **Pro Tips:**
• The AI learns from your preferences over time
• Use natural language - no need for exact commands
• Ask follow-up questions for deeper analysis
• Combine multiple requests: "crypto news and sentiment from last week"

🚀 **Advanced Queries:**
• "Compare sentiment between tech and climate news"
• "Summarize the main themes in today's political coverage"
• "What are people saying about the latest AI developments?"
• "Give me a comprehensive analysis of renewable energy news"

Type 'quit' or 'exit' to end the session.
"""


async def main():
    """Main function to run the enhanced interactive agent"""
    try:
        # Validate configuration
        validate_config()
        
        # Create and start enhanced agent
        agent = EnhancedMCPAgent()
        
        print("🚀 Starting Enhanced News Analysis Agent with LangChain...")
        session_started = await agent.start_session()
        
        if not session_started:
            print("Failed to start MCP session. Please check server configuration.")
            return
        
        print("🤖 Enhanced News Analysis Agent is ready!")
        print("🧠 Features: LLM-powered reasoning, conversation memory, smart analysis")
        print("Type 'quit' to exit, 'help' for example queries")
        print("-" * 60)
        
        # Interactive loop
        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print("\n🔄 Processing with AI reasoning...")
                response = await agent.process_query(query)
                
                if response['success']:
                    print(f"\n{response['response']}")
                    
                    # Show AI insights
                    if response.get('reasoning'):
                        print(f"\n🤖 **AI Reasoning:** {response['reasoning']}")
                    
                    if response.get('follow_up_suggestions'):
                        print(f"\n💡 **Suggestions:**")
                        for i, suggestion in enumerate(response['follow_up_suggestions'][:2], 1):
                            print(f"  {i}. {suggestion}")
                    
                    # Show metadata for debugging if needed
                    if logger.level <= logging.INFO:
                        intent = response.get('intent', 'unknown')
                        confidence = response.get('confidence', 0.0)
                        print(f"\n🎯 _Detected: {intent} (confidence: {confidence:.2f})_")
                        
                else:
                    error_msg = response.get('error', 'Unknown error')
                    print(f"\n❌ **Error:** {error_msg}")
                    print("💡 _Try rephrasing your query or type 'help' for examples_")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
        
        # Cleanup
        await agent.close_session()
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"Enhanced agent startup failed: {e}")
        logger.error(f"Enhanced agent startup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
