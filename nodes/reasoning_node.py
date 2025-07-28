import logging
import json
from typing import Any, Dict
from ..utilities.runner_utilities import find_matching_task
from ..models.configurable_agent_executor import State, Task
from src.services.llm.openai import OpenAIService
from src.services.llm.groq import GroqService
from src.services.llm.inflection import InflectionService

logger = logging.getLogger(__name__)

def _generate_reasoning_node_system_prompt(matching_task: Task, state: State, prompt_vars: Dict[str, Any]) -> str:
    """Generate the system prompt for the reasoning node."""
    try:
        # Create base system prompt with fallback
        base_prompt = matching_task.prompt_system if matching_task.prompt_system else "You are a helpful AI assistant."
        
        # Add JSON formatting instructions with proper escaping
        json_format = """
            Format your response as a valid JSON object with these exact keys:
            {{
                "reasoning": "Your step-by-step thought process",
                "response": "Your response"
            }}
            Ensure the response is valid JSON and can be parsed by json.loads().
            """
        
        # Combine base prompt with JSON formatting
        system_prompt = f"{base_prompt}\n{json_format}"

        # Format with variables if provided
        if prompt_vars:
            try:
                # First format the base prompt with variables
                if "{" in base_prompt and "}" in base_prompt:
                    # Validate prompt_vars are strings
                    safe_prompt_vars = {k: str(v) for k, v in prompt_vars.items()}
                    base_prompt = base_prompt.format(**safe_prompt_vars)
                
                # Then combine with the JSON format
                system_prompt = f"{base_prompt}\n{json_format}"
            except KeyError as e:
                logging.warning(f"Missing variable in system prompt: {e}")
            except Exception as e:
                logging.warning(f"Error formatting system prompt: {e}")
        
        return system_prompt
        
    except Exception as e:
        logging.error(f"Error generating system prompt: {e}")
        # Return a safe fallback prompt
        return """""You are a helpful AI assistant. Format your response as a valid JSON object with these exact keys:
            {{
                "reasoning": "Your step-by-step thought process",
                "response": "Your response"
        }}"""""

def _generate_reasoning_node_user_prompt(matching_task: Task, state: State, prompt_vars: Dict[str, Any]) -> str:
    """Generate the user prompt for the reasoning node."""
    user_prompt = f"""Process this task:
            Name: {matching_task.task_name}
            Description: {matching_task.task_description}
            """
    
    if prompt_vars:
        user_prompt += "\nInput Variables:\n"
        for var_name, var_value in prompt_vars.items():
            user_prompt += f"{var_name}: {var_value}\n"
    
    if matching_task.prompt_user:
        try:
            custom_prompt = matching_task.prompt_user.format(**prompt_vars)
            custom_prompt = custom_prompt.format(**state)
            user_prompt += f"\nCustom Instructions:\n{custom_prompt}"
        except KeyError as e:
            logging.warning(f"Could not format custom user prompt. Missing variable: {e}")
    
    return user_prompt

def process_groq_response(content: str) -> dict:
    """
    Process the Groq model's response content, handling JSON parsing and cleaning.
    
    Args:
        content (str): The raw content from the Groq model's response
        
    Returns:
        dict: The parsed JSON output from the model
    """
    try:
        # Validate content
        if not content:
            logger.error("Received empty content from Groq model")
            raise ValueError("Empty content received from Groq model")
            
        logger.debug(f"Raw Groq model response content type: {type(content)}")
        logger.debug(f"Raw Groq model response content length: {len(content)}")
        logger.debug(f"Raw Groq model response content: {repr(content)}")
        
        # Remove 'json' prefix if present
        if content.lower().startswith('json'):
            content = content[4:].strip()
            logger.debug(f"Removed 'json' prefix. New content: {repr(content)}")
        
        # Try to find the first '{' and last '}'
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            logger.error("No JSON object markers found in content")
            raise ValueError("No JSON object markers found in content")
            
        # Extract just the JSON object
        json_content = content[start_idx:end_idx + 1]
        logger.debug(f"Extracted JSON content: {repr(json_content)}")
        
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from Groq model: {str(e)}")
        logger.error(f"Raw content that failed to parse: {repr(content)}")
        
        # Try to clean the content if it's a string
        if isinstance(content, str):
            try:
                # Remove 'json' prefix if present and clean the content
                if content.lower().startswith('json'):
                    content = content[4:].strip()
                cleaned_content = content.strip().encode('utf-8').decode('utf-8-sig')
                logger.debug(f"Cleaned content: {repr(cleaned_content)}")
                
                # Try to find the first '{' and last '}'
                start_idx = cleaned_content.find('{')
                end_idx = cleaned_content.rfind('}')
                
                if start_idx == -1 or end_idx == -1:
                    logger.error("No JSON object markers found in cleaned content")
                    raise ValueError("No JSON object markers found in cleaned content")
                    
                # Extract just the JSON object
                json_content = cleaned_content[start_idx:end_idx + 1]
                logger.debug(f"Extracted JSON content from cleaned content: {repr(json_content)}")
                
                return json.loads(json_content)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON even after cleaning: {str(e2)}")
                raise
        else:
            raise

def process_openai_response(content: str) -> dict:
    """
    Process the OpenAI model's response content, handling JSON parsing and cleaning.
    
    Args:
        content (str): The raw content from the OpenAI model's response
        
    Returns:
        dict: The parsed JSON output from the model
    """
    try:
        logger.debug(f"Raw OpenAI model response content: {content}")
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from OpenAI model: {str(e)}")
        logger.error(f"Raw content that failed to parse: {repr(content)}")
        
        # Try to clean the content if it's a string
        if isinstance(content, str):
            try:
                cleaned_content = content.strip()
                logger.debug(f"Cleaned content: {repr(cleaned_content)}")
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON even after cleaning: {str(e2)}")
                raise
        else:
            raise

def process_inflection_response(content: str) -> dict:
    """
    Process the Inflection model's response content, handling JSON parsing and cleaning.
    
    Args:
        content (str): The raw content from the Inflection model's response
        
    Returns:
        dict: The parsed JSON output from the model
    """
    try:
        logger.debug(f"Raw Inflection model response content: {content}")
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from Inflection model: {str(e)}")
        logger.error(f"Raw content that failed to parse: {repr(content)}")
        
        # Try to clean the content if it's a string
        if isinstance(content, str):
            try:
                cleaned_content = content.strip()
                logger.debug(f"Cleaned content: {repr(cleaned_content)}")
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON even after cleaning: {str(e2)}")
                raise
        else:
            raise

async def async_reasoning_node(agent, state: State) -> Dict[str, Any]:
    """
    Internal async implementation of the reasoning node.
    """
    try:
        # Validate state has required attributes
        if not isinstance(state, dict):
            raise ValueError(f"State must be a dictionary, got {type(state)}")
            
        # Get the matching task
        matching_task = find_matching_task(state)

        # Prepare prompt variables
        prompt_vars = {}
        for var in matching_task.input_variables:
            var_name = var['name']
            if var_name in state["data_object"]:
                prompt_vars[var_name] = state["data_object"][var_name]
            else:
                logging.warning(f"Warning: Required input variable {var_name} not found in state")

        # Generate prompts
        system_prompt = _generate_reasoning_node_system_prompt(matching_task, state, prompt_vars)
        user_prompt = _generate_reasoning_node_user_prompt(matching_task, state, prompt_vars)

        # Parse model_id to get provider and model name
        model_parts = matching_task.model_id.split(":")
        provider = model_parts[0] if len(model_parts) > 1 else None
        model_name = model_parts[1] if len(model_parts) > 1 else matching_task.model_id

        # Create messages for the chat completion
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # Use OpenAIService for OpenAI models
        if provider == "openai":
            openai_service = OpenAIService(
                model_name=model_name,
                temperature=0.2,
                max_tokens=1024
            )
            response = await openai_service.chat_completion(messages)
        elif provider == "groq":
            groq_service = GroqService(
                model_name=model_name,
                temperature=0.2,
                max_tokens=1024
            )
            response = await groq_service.chat_completion(messages)
        elif provider == "inflection":
            inflection_service = InflectionService(
                model_name=model_name,
                temperature=0.2,
                max_tokens=1024
            )
            response = await inflection_service.chat_completion(messages)
        else:
            # Log warning for unsupported model
            logger.warning(f"Model provider '{provider}' is not supported. Only 'openai', 'groq', and 'inflection' providers are currently supported.")
            raise ValueError(f"Model provider '{provider}' is not supported. Only 'openai', 'groq', and 'inflection' providers are currently supported.")
        
        # Parse the JSON response
        try:
            content = response.choices[0].message.content
            if provider == "openai":
                model_output = process_openai_response(content)
            elif provider == "groq":
                model_output = process_groq_response(content)
            elif provider == "inflection":
                model_output = process_inflection_response(content)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            model_reasoning = model_output.get("reasoning", "No reasoning provided")
            model_response = model_output.get("response", "No response provided")
        except json.JSONDecodeError:
            content = response.choices[0].message.content
            model_reasoning = "Error: Could not parse JSON response"
            model_response = content

        # Return updated state attributes with the Task object and parsed response
        output_var_name = matching_task.output_variables[0]["name"]
        updated_data_object = agent._set_dict_attribute(
            state.get("data_object", {}),
            output_var_name,
            model_response
        )
        
        return {
            "data_object": updated_data_object,
            "model_reasoning": model_reasoning,
            "next_task": matching_task.output_dependencies[0] if matching_task.output_dependencies else None
        }

    except Exception as e:
        logging.error(f"Error processing task in reasoning node: {str(e)}")
        raise ValueError(f"Error processing task in reasoning node: {str(e)}")
