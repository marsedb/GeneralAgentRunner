import logging
import json
import os
#from typing_extensions import TypedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable

from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.schemas.agent import AgentResult
from src.agents.base.agent_base import (
    AgentDependencies,
    AgentResult,
    BaseAgent,
)
from src.schemas.agent import AgentMetadata, AgentResult
from src.agents.base.agent_base import AgentMcpServers
from openai import OpenAI
from .models.configurable_agent_executor import Task, State, CurrentAgent
from .nodes.reasoning_node import async_reasoning_node
from .nodes.tool_node import async_tool_node
from .utilities.ui_utilities import generate_response_and_ui_components
from .utilities.runner_utilities import (
    initialize_state,
    load_state,
    save_state,
    clear_cache,
    prepare_conversation_context,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfigurableAgentExecutorDependencies:
    """Dependencies for the configurable agent executor"""
    config_file_path: Optional[str] = None


class ConfigurableAgentExecutor(BaseAgent):
    """Agent for handling general queries with configurable workflow"""

    def __init__(self, name: str, agent_id: int):
        self._workflow = None
        self._task_list = None
        self.name: str = name
        self.agent_metadata: Optional[AgentMetadata] = None
        self.user_id: Optional[int] = None
        self.mcp_servers: Optional[AgentMcpServers] = None
        self.agent_id: int = agent_id
    
    def _get_config_json(self) -> Dict[str, Any]:
        """Reads and returns JSON configuration from either the database or config file."""
        # First try to get configuration from database
        if self.agent_id:
            try:
                from src.models import AgentConfiguration
                from src.db.database import get_db
                
                db = next(get_db())
                config = db.query(AgentConfiguration).filter(
                    AgentConfiguration.agent_id == self.agent_id
                ).first()
                
                if config and config.configuration:
                    logger.info(f"Found configuration in database for agent {self.agent_id}")
                    return config.configuration
            except Exception as e:
                logger.warning(f"Error retrieving configuration from database: {e}")
        
        
    def _parse_tasks_from_config(self, config_data: Dict[str, Any]) -> List[Task]:
        """Parse the configuration data and create a list of Task objects."""
        tasks = []
        
        try:
            enriched_tasks = config_data.get("enriched_tasks", [])
            
            for task_data in enriched_tasks:
                tool_parameters = task_data["tool_parameters"]
                if isinstance(tool_parameters, str):
                    tool_parameters = json.loads(tool_parameters)
                    
                task = Task(
                    id=task_data["id"],
                    input_dependencies=task_data["input_dependencies"],
                    output_dependencies=task_data["output_dependencies"],
                    task_name=task_data["task_name"],
                    task_description=task_data["task_description"],
                    tool_id=task_data["tool_id"],
                    tool_action=task_data["tool_action"],
                    tool_parameters=tool_parameters,
                    model_id=task_data["model_id"],
                    model_reason=task_data["model_reason"],
                    input_variables=task_data["input_variables"],
                    output_variables=task_data["output_variables"],
                    prompt_system=task_data["prompt_system"],
                    prompt_user=task_data["prompt_user"],
                    prompt_variables=task_data["prompt_variables"]
                )
                tasks.append(task)
                
        except KeyError as e:
            raise KeyError(f"Missing required field in configuration: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid tool_parameters JSON string: {str(e)}")
        
        return tasks
   
    def _set_dict_attribute(self, dictionary: dict, attribute_name: str, value: Any) -> dict:
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

    def _build_task_workflow(self, tasks: List[Task]) -> StateGraph:
        """
        Builds a LangGraph workflow from a list of tasks.
        
        Args:
            tasks (List[Task]): List of Task objects to build the workflow from
            
        Returns:
            StateGraph: Compiled workflow with nodes for each task
        """
        try:
        # Initialize the graph
            workflow = StateGraph(State)

            #  a small factory that closes over `self`
            def make_node(handler: Callable):
                async def node(state):
                # handler is either async_tool_node or async_reasoning_node
                   return await handler(self, state)
                return node

            # Add nodes for each task
            for task in tasks:
                # Use tool_node if task has a tool_id, otherwise use reasoning_node
                if task.tool_id:  
                    node_fn = make_node(async_tool_node)
                else:
                    node_fn = make_node(async_reasoning_node)
                workflow.add_node(f"node_{task.id}", node_fn)

            # Add edges based on task dependencies
            for task in tasks:
                if not task.input_dependencies:
                    # If no input dependencies, connect to START
                    workflow.add_edge(START, f"node_{task.id}")
                
                if not task.output_dependencies:
                    # If no output dependencies, connect to END
                    workflow.add_edge(f"node_{task.id}", END)
                else:
                    # Add edges to dependent tasks
                    for dep in task.output_dependencies:
                        workflow.add_edge(f"node_{task.id}", f"node_{dep}")
            
            return workflow
        
        except Exception as e:
            logging.error(f"Error building task workflow: {e}")
            raise e 

    def build_data_object_from_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """
        Creates a data object dictionary by collecting all input and output variable names from the tasks.
        
        Args:
            tasks (List[Task]): List of Task objects to process
            
        Returns:
            Dict[str, Any]: Dictionary with keys for each input and output variable name, initialized to None
            
        Example:
            If tasks have:
            input_variables: ['topic', 'joke']
            output_variables: ['joke', 'improved_joke', 'final_joke']
            returns {'topic': None, 'joke': None, 'improved_joke': None, 'final_joke': None}
        """
        try:
            data_object = {}
        
            for task in tasks:
                # Process input variables
                if task.input_variables:
                    for var in task.input_variables:
                        var_name = var['name']
                        if var_name not in data_object:
                            data_object[var_name] = None
                
                # Process output variables
                if task.output_variables:
                    for var in task.output_variables:
                        var_name = var['name']
                        if var_name not in data_object:
                            data_object[var_name] = None
            
                        # Set results_variable from last task's output variables if they exist
            if tasks and tasks[-1].output_variables:
                last_task = tasks[-1]
                if last_task.output_variables:
                    # Get the first output variable name from the last task
                    results_var_name = last_task.output_variables[0]["name"]
                    data_object["results_variable"] = results_var_name
 
            return data_object
        
        except Exception as e:
            logging.error(f"Error building data object from tasks: {e}")
            raise ValueError(f"Error building data object from tasks: {e}")

    async def run(
        self,
        query: str,
        deps: Optional[ConfigurableAgentExecutorDependencies] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
    ) -> AgentResult:
        """
        Run the configurable agent with workflow execution.

        Args:
            query: The user's query
            deps: Optional dependencies including config file path
            conversation_history: Optional list of previous messages
            db: The database session
            user_id: The user's ID

        Returns:
            AgentResult with the response

        Raises:
            RuntimeError: If state loading fails
        """
        try:
            final_state = None
            # Load state 
            current_agent = load_state(self, deps, user_id)
            if not current_agent:
                raise RuntimeError(f"Failed to load state for agent {self.user_id}-{self.name}")

            # Build workflow if not already done
            if not current_agent["workflow"]:
                current_agent["workflow"] = self._build_task_workflow(current_agent["task_list"])

            # Initialize state
            current_agent["data_objects"]["user_query"] = prepare_conversation_context(query, conversation_history)
            # Create properly typed initial state
            initial_state: State = {
                "task_list": current_agent["task_list"],
                "next_task": current_agent["task_list"][0].id if current_agent["task_list"] else "",  # Safely get first task ID
                "data_object": current_agent["data_objects"]  # Add the data object to the initial state
            }
                
            # Execute workflow
            #chain = current_agent["workflow"].compile()

            try:
                # Run the workflow synchronously
                #final_state = await chain.ainvoke(initial_state)
                graph = current_agent["workflow"].compile()
                final_state = await graph.ainvoke(initial_state)
                
                response = final_state["data_object"].get(
                    final_state["data_object"].get("results_variable", "response"),
                    "No response generated"
                )
                
                # Convert response to string if it's a dictionary
                if isinstance(response, dict):
                    response = str(response)
                
                if current_agent["render_ui"]:
                    chat_response, ui_components = await generate_response_and_ui_components(response, query)
                else:
                    chat_response = response
                    ui_components = []
                    
                return AgentResult(
                    response=chat_response,
                    data={"query": query},
                    ui_components= ui_components
                )
            except Exception as e:
                logging.error(f"Error executing workflow: {str(e)}")
                return AgentResult(
                    response="I apologize, but I encountered an error while processing your request.",
                    data={"query": query, "error": str(e)}
                )

        except Exception as e:
            logging.error(f"Error in configurable agent: {e}")
            return AgentResult(
                response="I apologize, but I'm having trouble processing your request at the moment. Please try again later.",
                data={"query": query, "error": str(e)}
            )
        finally:
            # If done then clear object
            if final_state is None or final_state.get("next_task") is None:
                clear_cache(self, user_id)
            else:
                # Save state before returning
                save_state(current_agent, user_id)


 
