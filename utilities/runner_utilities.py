import logging
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from ..models.configurable_agent_executor import Task, State
from ..models.configurable_agent_executor import CurrentAgent

logger = logging.getLogger(__name__)

def set_dict_attribute(dictionary: dict, attribute_name: str, value: Any) -> dict:
    """
    Set a dictionary attribute to a given value.
    
    Args:
        dictionary: The dictionary to modify
        attribute_name: The name of the attribute to set
        value: The value to set the attribute to
        
    Returns:
        The modified dictionary
    """
    dictionary[attribute_name] = value
    return dictionary

def initialize_state(agent, user_id: Optional[int] = None) -> CurrentAgent:
    """
    Initialize all required state attributes with default values.
    Called when no cached state is found.
    
    Args:
        agent: The agent instance to initialize
        config_file_path: Optional path to config file
        
    Returns:
        CurrentAgent: The initialized agent state
    """
    # Initialize required attributes with default values
    try:
        config_json = agent._get_config_json()
        task_list = agent._parse_tasks_from_config(config_json)
        render_ui = should_render_ui(config_json)
        
        # Create a new CurrentAgent with all required fields
        current_agent: CurrentAgent = {
            "thread_id": None,
            "config_json": config_json,
            "workflow": None,
            "task_list": task_list,
            "data_objects": agent.build_data_object_from_tasks(task_list),
            "execution_history": [],
            "next_step": None,
            "name": agent.name,
            "agent_metadata": agent.agent_metadata,
            "user_id": user_id,
            "mcp_servers": agent.mcp_servers,
            "agent_id": agent.agent_id,
            "render_ui": render_ui
        }
        
        logging.info(f"Initialized new state for agent {agent.user_id}-{agent.name}")
        return current_agent
        
    except Exception as e:
        logging.error(f"Error initializing state for agent {agent.user_id}-{agent.name}: {e}")
        return None

def load_state_from_cache_object(agent: State, state_data: dict, cache_key: str) -> CurrentAgent:
    """
    Load state attributes from a cache object.
    
    Args:
        agent: The agent instance to load state into
        state_data: The state data loaded from cache
        cache_key: The cache key used for logging
        
    Returns:
        CurrentAgent: The loaded agent state
    """
    from typing import Dict, List
        
    # Create a new CurrentAgent dictionary with default values
    current_agent: CurrentAgent = {
        "thread_id": None,
        "config_json": None,
        "workflow": None,
        "task_list": None,
        "data_objects": {},
        "execution_history": [],
        "next_step": None,
        "name": agent.get("name", ""),
        "agent_metadata": agent.get("agent_metadata"),
        "user_id": agent.get("user_id"),
        "mcp_servers": agent.get("mcp_servers"),
        "agent_id": agent.get("agent_id", 0)
    }
    
    # Update values from state_data if they exist
    for key in state_data:
        if key in current_agent:
            if key == '_task_list' and isinstance(state_data[key], list):
                # Convert dictionaries back to Task objects
                current_agent["task_list"] = [
                    Task(
                        id=task_dict['id'],
                        input_dependencies=task_dict['input_dependencies'],
                        output_dependencies=task_dict['output_dependencies'],
                        task_name=task_dict['task_name'],
                        task_description=task_dict['task_description'],
                        tool_id=task_dict['tool_id'],
                        tool_action=task_dict['tool_action'],
                        tool_parameters=task_dict['tool_parameters'],
                        model_id=task_dict['model_id'],
                        model_reason=task_dict['model_reason'],
                        input_variables=task_dict['input_variables'],
                        output_variables=task_dict['output_variables'],
                        prompt_system=task_dict['prompt_system'],
                        prompt_user=task_dict['prompt_user'],
                        prompt_variables=task_dict['prompt_variables']
                    ) for task_dict in state_data[key]
                ]
            else:
                current_agent[key] = state_data[key]
    
    logging.info(f"Successfully loaded state for user-agent {cache_key}")
    return current_agent

def load_state(agent, deps, user_id) -> CurrentAgent:
    """
    Load agent state from cache.
    
    Args:
        agent: The agent instance
        deps: Optional dependencies including config file path
        
    Returns:
        CurrentAgent: The loaded or initialized agent state
    """
    try:
        from src.services.cache import get_redis_cache
        
        # Get cache instance
        cache = get_redis_cache()
        # Create cache key using user_id and agent name
        cache_key = f"{user_id}-{agent.agent_id}"
        
        # Get state from cache
        state_data = cache.get_json(cache_key)
        if not state_data:
            # Initialize new state if no cached state exists
            return initialize_state(agent, user_id)
        
        # Load state from cache object
        return load_state_from_cache_object(agent, state_data, cache_key)
        
    except Exception as e:
        logging.error(f"Error loading state for user-agent {cache_key}: {e}")
        return False

def save_state(current_agent: CurrentAgent) -> bool:
    """
    Save agent state to cache.
    
    Args:
        current_agent: The CurrentAgent dictionary to save
        
    Returns:
        bool: True if state was saved successfully, False otherwise
    """
    try:
        from src.services.cache import get_redis_cache
        
        # Get cache instance
        cache = get_redis_cache()
        
        # Create cache key using user_id and agent name
        cache_key = f"{current_agent['user_id']}-{current_agent['agent_id']}"
        
        # Convert to dictionary, excluding non-serializable objects
        state_data = {}
        for key, value in current_agent.items():
            # Skip private methods and attributes except specific ones we want to save
            if key.startswith('_') and key not in ['_task_list']:
                continue
                
            # Special handling for task_list
            if key == 'task_list' and value is not None:
                # Convert Task objects to dictionaries
                state_data[key] = [asdict(task) for task in value]
                continue

            # Special handling for data_objects
            if key == 'data_objects' and value is not None:
                # Convert data_objects to a serializable format
                try:
                    state_data[key] = json.loads(json.dumps(value))
                    continue
                except (TypeError, OverflowError) as e:
                    logging.warning(f"Error serializing data_objects: {e}")
                    continue
                
            # Skip non-serializable objects
            try:
                # Test if the value is JSON serializable
                json.dumps({key: value})
                state_data[key] = value
            except (TypeError, OverflowError):
                logging.warning(f"Skipping non-serializable attribute {key}")
                continue
        
        # Save state to cache
        if cache.set_json(cache_key, state_data):
            logging.info(f"Successfully saved state for user-agent {cache_key}")
            return True
        else:
            logging.error(f"Failed to save state for user-agent {cache_key}")
            return False
            
    except Exception as e:
        logging.error(f"Error saving state for user-agent {cache_key}: {e}")
        return False

def clear_cache(self, user_id: int) -> bool:
    """
    Clear the agent's cached state from Redis.
    
    Args:
        agent: The agent instance
        
    Returns:
        bool: True if cache was cleared successfully, False otherwise
    """
    try:
        from src.services.cache import get_redis_cache
        
        # Get cache instance
        cache = get_redis_cache()
        
        # Create cache key using user_id and agent name
        cache_key = f"{user_id}-{self.agent_id}"
        
        # Delete the cache entry
        if cache.delete(cache_key):
            logging.info(f"Successfully cleared cache for user-agent {cache_key}")
            # Reinitialize the state
            initialize_state(self, user_id)
            return True
        else:
            #logging.error(f"Failed to clear cache for user-agent {cache_key}")
            #Cache object may not exist
            initialize_state(self, user_id)
            return False
            
    except Exception as e:
        logging.error(f"Error clearing cache for user-agent {cache_key}: {e}")
        return False

def find_matching_task(state: State) -> Task:
    """Find the task in the task list that matches the next_task_id."""
    if 'next_task' not in state:
        raise ValueError("State missing 'next_task' attribute")
    if 'task_list' not in state:
        raise ValueError("State missing 'task_list' attribute")

    next_task_id = state['next_task']
    task_list = state['task_list']

    matching_task = next((task for task in task_list if task.id == next_task_id), None)
    
    if not matching_task:
        raise ValueError(f"No task found with id '{next_task_id}' in task_list")
        
    return matching_task


def print_tasks(tasks: List[Task]):
    """
    Process the configuration file and create task objects.
    
        Returns:
            List[Task]: List of Task objects
        """
        # Example: Print task information
    for task in tasks:
                print("\n" + "="*50)
                print(f"Task: {task.task_name}")
                print(f"ID: {task.id}")
                print(f"Description: {task.task_description}")
                print(f"Tool: {task.tool_id} - {task.tool_action}")
                print(f"Tool Parameters: {task.tool_parameters}")
                print(f"Dependencies:")
                print(f"  Input: {task.input_dependencies}")
                print(f"  Output: {task.output_dependencies}")
                print(f"Variables:")
                print(f"  Input: {task.input_variables}")
                print(f"  Output: {task.output_variables}")
                if task.model_id:
                    print(f"Model ID: {task.model_id}")
                if task.model_reason:
                    print(f"Model Reason: {task.model_reason}")
                if task.prompt_system:
                    print(f"System Prompt: {task.prompt_system}")
                if task.prompt_user:
                    print(f"User Prompt: {task.prompt_user}")
                if task.prompt_variables:
                    print(f"Prompt Variables: {task.prompt_variables}")  
    return tasks

def prepare_conversation_context(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Prepare conversation context by combining recent conversation history with the current query.
    
    Args:
        query: The current user query
        conversation_history: Optional list of previous messages
        
    Returns:
        str: The combined query with conversation context
    """
    try:
        context = ""
        if conversation_history and len(conversation_history) > 1:
            # Get the last few exchanges, limited to 5 messages
            recent_history = (
                conversation_history[-5:]
                if len(conversation_history) > 5
                else conversation_history
            )
            context = "\nConversation context:\n"
            for msg in recent_history:
                context += f"{msg['role']}: {msg['content']}\n"

        return f"User query: {query}{context}"
    except Exception as e:
        logging.error(f"Error preparing conversation context: {e}")
        return f"User query: {query}"

def should_render_ui(data: Dict[str, Any]) -> bool:
    """
    Check if UI should be rendered based on the render_ui attribute in the data.
    
    Args:
        data: Dictionary containing the data to check
        
    Returns:
        bool: False if render_ui is explicitly set to False, True otherwise or if any error occurs
    """
    try:
        return data.get("render_ui", True) is not False
    except Exception as e:
        logging.warning(f"Missing render_ui attribute: {str(e)}")
        return True

