from typing_extensions import TypedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from src.schemas.agent import AgentMetadata, AgentResult
from src.agents.base.agent_base import AgentMcpServers


@dataclass
class Task:
    id: str
    input_dependencies: List[str]
    output_dependencies: List[str]
    task_name: str
    task_description: str
    tool_id: str
    tool_action: str
    tool_parameters: Dict[str, Any]
    model_id: Optional[str]
    model_reason: Optional[str]
    input_variables: List[Dict[str, Any]]
    output_variables: List[Dict[str, Any]]
    prompt_system: Optional[str]
    prompt_user: Optional[str]
    prompt_variables: List[Dict[str, Any]]


class State(TypedDict):
    data_object: Dict[str, Any]
    task_list: List[Task]
    next_task: str 

class CurrentAgent(TypedDict):
    thread_id: Optional[str] = None
    config_json: Optional[str] = None
    workflow: Optional[StateGraph] = None
    task_list: Optional[List[Task]] = None
    data_objects: Optional[Dict[str, Any]] = None
    execution_history: Optional[List[Dict[str, Any]]] = None
    next_step: Optional[str] = None
    name: str
    agent_metadata: Optional[AgentMetadata] = None
    user_id: Optional[int] = None
    mcp_servers: Optional[AgentMcpServers] = None
    agent_id: int
    render_ui: Optional[bool] = True