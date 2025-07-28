import os
import logging
from typing import Optional, Dict, Any
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.settings import ModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.anthropic import AnthropicModel

logger = logging.getLogger(__name__)

def get_model_provider(model_id: str, provider: str) -> Any:
    """
    Get the appropriate model provider based on the model ID and provider.
    
    Args:
        model_id: The ID of the model to use
        provider: The provider to use (e.g., 'openai', 'groq', 'anthropic')
        
    Returns:
        Model instance for the specified model ID and provider
    """
    try:
        # Get API key based on provider
        if provider.lower() == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            provider_instance = OpenAIProvider(api_key=api_key)
            return OpenAIModel(model_id, provider=provider_instance)
            
        elif provider.lower() == 'groq':
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            provider_instance = GroqProvider(api_key=api_key)
            return GroqModel(model_id, provider=provider_instance)
            
        elif provider.lower() == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            provider_instance = AnthropicProvider(api_key=api_key)
            return AnthropicModel(model_id, provider=provider_instance)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        logger.error(f"Error getting model provider: {e}")
        raise

async def run_model(
    model_id: str,
    provider: str,
    message: str,
    system_prompt: str,
    target_server: Any,
    temperature: float = 0,
    max_tokens: int = 512,
    tool_choice: str = "required"
) -> str:
    """
    Run a model with the specified parameters.
    
    Args:
        model_id: The ID of the model to use
        provider: The provider to use (e.g., 'openai', 'groq', 'anthropic')
        message: The user message to process
        system_prompt: The system prompt to use
        target_server: The target MCP server
        temperature: Model temperature setting
        max_tokens: Maximum tokens to generate
        tool_choice: Tool choice setting
        
    Returns:
        str: The model's response
    """
    try:
        logger.debug(f"Running tool node with model {model_id} with provider {provider}")
        model = get_model_provider(model_id, provider)
        
        agent_instance = Agent(
            model,
            mcp_servers=[target_server],
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                temperature=temperature,
                tool_choice=tool_choice
            ),
        )
        
        async with agent_instance.run_mcp_servers():
            response = await agent_instance.run(
                message,
                usage_limits=UsageLimits(request_limit=5)
            )
            
        return response
        
    except Exception as e:
        logger.error(f"Error running model {model_id}: {e}")
        raise 