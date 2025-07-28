import logging
import json
import os
from typing import Any, Dict
from openai import OpenAI

from ..models.configurable_agent_executor import State, Task
from pydantic_ai.mcp import MCPServerHTTP
from ..utilities.runner_utilities import find_matching_task
from ..utilities.model_utilities import run_model

logger = logging.getLogger(__name__)

def generate_tool_node_system_prompt(task: Task, state: State, tool_params: Dict[str, Any], user_tokens: str) -> str:
    """Generate the system prompt for the tool node."""
    try:
        # Create base system prompt with fallback
        base_prompt = task.prompt_system if task.prompt_system else "You are a helpful AI assistant with access to tools."
        
        # Add JSON formatting instructions with proper escaping
        #response_format = 
        """
            Format your response as a valid JSON object with these exact keys:
            {{
                "reasoning": "Your step-by-step thought process",
                "response": "Your response"
            }}
            Ensure the response is valid JSON and can be parsed by json.loads().
            """
        response_format = """ format your response as a user friendly response that can be displayed to the user. Do not include any other text in your response."""

        user_tokens_prompt = f"Pass the following user tokens to the tool: {user_tokens}"
        # Combine base prompt with JSON formatting
        system_prompt = f"{base_prompt}\n{response_format}\n{user_tokens_prompt}"

        # Format with variables if provided
        if tool_params:
            try:
                # First format the base prompt with variables
                if "{" in base_prompt and "}" in base_prompt:
                    # Validate tool_params are strings
                    safe_tool_params = {k: str(v) for k, v in tool_params.items()}
                    base_prompt = base_prompt.format(**safe_tool_params)
                
                # Then combine with the JSON format
                system_prompt = f"{base_prompt}\n{response_format}\n{user_tokens_prompt}"
            except KeyError as e:
                logging.warning(f"Missing variable in system prompt: {e}")
            except Exception as e:
                logging.warning(f"Error formatting system prompt: {e}")
        
        return system_prompt
        
    except Exception as e:
        logging.error(f"Error generating system prompt: {e}")
        # Return a safe fallback prompt
        return """You are a helpful AI assistant. Format your response as a valid JSON object with these exact keys:
            {{
                "reasoning": "Your step-by-step thought process",
                "response": "Your response"
            }}"""

def generate_tool_node_user_prompt(task: Task, state: State, tool_params: Dict[str, Any]) -> str:
    """Generate the user prompt for the tool node."""
    user_prompt = f"""Process this task:
            Name: {task.task_name}
            Description: {task.task_description}
            """
    
    if tool_params:
        user_prompt += "\nInput Variables:\n"
        for var_name, var_value in tool_params.items():
            user_prompt += f"{var_name}: {var_value}\n"
    
    if task.prompt_user:
        try:
            custom_prompt = task.prompt_user.format(**tool_params)
            custom_prompt = custom_prompt.format(**state)
            user_prompt += f"\nCustom Instructions:\n{custom_prompt}"
        except KeyError as e:
            logging.warning(f"Could not format custom user prompt. Missing variable: {e}")
    
    return user_prompt

async def async_tool_node(agent, state: State) -> Dict[str, Any]:
    """
    Internal async implementation of the tool node.
    """
    try:
        # Validate state has required attributes
        if not isinstance(state, dict):
            raise ValueError(f"State must be a dictionary, got {type(state)}")
            
        # Get the matching task
        matching_task = find_matching_task(state)

        # Prepare tool parameters
        tool_params = {}
        for var in matching_task.input_variables:
            var_name = var['name']
            if var_name in state["data_object"]:
                tool_params[var_name] = state["data_object"][var_name]
            else:
                logging.warning(f"Warning: Required input variable {var_name} not found in state")

        if not agent.mcp_servers:
            raise ValueError("No MCP servers configured for this agent")

        # Find the appropriate MCP server for this tool
        mcp_server = None
        for server in agent.mcp_servers.mcp_servers:
            if server.mcp_server_id == int(matching_task.tool_id):
                mcp_server = server
                break

        if not mcp_server:
            raise ValueError(f"No MCP server found for tool ID {matching_task.tool_id}")

        try:
            target_server = MCPServerHTTP(
                url=mcp_server.mcp_server_url,
                headers={},
                timeout=30,
                sse_read_timeout=60,
            )
            user_tokens = mcp_server.encrypted_user_tokens

            system_prompt = generate_tool_node_system_prompt(matching_task, state, tool_params, user_tokens)
            user_prompt = generate_tool_node_user_prompt(matching_task, state, tool_params)

            # Get model settings from task
            # Parse model_id to get provider and model name
            model_parts = matching_task.model_id.split(":")
            provider = model_parts[0] if len(model_parts) > 1 else None
            model_name = model_parts[1] if len(model_parts) > 1 else matching_task.model_id

            model_id = model_name
            temperature = 0
            max_tokens = 512
            tool_choice = "required"
            
            # Use the model utilities to run the model
            response = await run_model(
                model_id=model_id,
                provider=provider,
                message=user_prompt,
                system_prompt=system_prompt,
                target_server=target_server,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice
            )

            if response and hasattr(response, "output"):
                tool_result = response.output

                output_var_name = matching_task.output_variables[0]["name"]
                updated_data_object = agent._set_dict_attribute(
                    state.get("data_object", {}),
                    output_var_name,
                    tool_result 
                )
                        
                return {
                    "data_object": updated_data_object,
                    "tool_result": tool_result,
                    "next_task": matching_task.output_dependencies[0] if matching_task.output_dependencies else None
                }
            else:
                error_text = await response.text()
                raise ValueError(f"MCP server returned status {response.status}: {error_text}")

        except Exception as e:
            print(f"Error running agent: {e}")         

    except Exception as e:
        logging.error(f"Error processing task in tool node: {str(e)}")
        raise ValueError(f"Error processing task in tool node: {str(e)}")

