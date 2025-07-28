# Agent Runner Module

The Agent Runner module is a configurable workflow execution system that allows for dynamic task-based AI agent processing. It provides a flexible framework for building complex AI workflows using LangGraph, with support for both reasoning nodes and tool-based execution.

## Overview

This module implements a configurable agent executor that can:
- Parse task configurations from JSON or database sources
- Build dynamic workflows using LangGraph
- Execute tasks with dependencies and data flow
- Support both reasoning (LLM-based) and tool-based nodes
- Integrate with MCP (Model Context Protocol) servers
- Generate UI components for different response types
- Maintain state across workflow executions

## File Structure

```
agent_runner/
├── configurable_agent_executor.py    # Main agent executor class
├── models/
│   └── configurable_agent_executor.py  # Data models and types
├── nodes/
│   ├── reasoning_node.py             # LLM-based reasoning nodes
│   └── tool_node.py                  # Tool execution nodes
└── utilities/
    ├── model_utilities.py            # Model provider utilities
    ├── runner_utilities.py           # State management utilities
    └── ui_utilities.py               # UI component generation
```

## Core Components

### 1. ConfigurableAgentExecutor (`configurable_agent_executor.py`)

The main agent class that orchestrates workflow execution.

**Key Features:**
- Loads configuration from database or JSON files
- Parses task definitions into executable workflows
- Builds LangGraph workflows dynamically
- Manages state persistence and caching
- Handles conversation context and user queries

**Main Methods:**
- `run()`: Main execution entry point
- `_get_config_json()`: Retrieves configuration from database
- `_parse_tasks_from_config()`: Converts config to Task objects
- `_build_task_workflow()`: Creates LangGraph workflow from tasks
- `build_data_object_from_tasks()`: Creates data structure for workflow

### 2. Data Models (`models/configurable_agent_executor.py`)

Defines the core data structures used throughout the system.

**Key Classes:**
- `Task`: Represents a single workflow task with all its properties
- `State`: TypedDict for workflow state management
- `CurrentAgent`: TypedDict for agent state persistence

**Task Properties:**
- `id`: Unique task identifier
- `input_dependencies`/`output_dependencies`: Task dependency management
- `tool_id`/`tool_action`: Tool execution parameters
- `model_id`/`model_reason`: LLM configuration
- `input_variables`/`output_variables`: Data flow specification
- `prompt_system`/`prompt_user`: Prompt templates

### 3. Execution Nodes

#### Reasoning Node (`nodes/reasoning_node.py`)

Handles LLM-based reasoning tasks without external tools.

**Features:**
- Supports multiple LLM providers (OpenAI, Groq, Inflection)
- Generates structured JSON responses
- Handles prompt templating with variables
- Processes conversation context
- Error handling and response cleaning

**Key Functions:**
- `async_reasoning_node()`: Main reasoning node implementation
- `_generate_reasoning_node_system_prompt()`: Creates system prompts
- `_generate_reasoning_node_user_prompt()`: Creates user prompts
- Provider-specific response processors

#### Tool Node (`nodes/tool_node.py`)

Executes tasks that require external tools via MCP servers.

**Features:**
- Integrates with MCP (Model Context Protocol) servers
- Supports multiple model providers
- Handles tool parameter formatting
- Manages server connections and authentication
- Error handling for tool execution

**Key Functions:**
- `async_tool_node()`: Main tool node implementation
- `generate_tool_node_system_prompt()`: Creates tool-specific prompts
- `generate_tool_node_user_prompt()`: Formats user prompts for tools

### 4. Utilities

#### Model Utilities (`utilities/model_utilities.py`)

Provides abstraction layer for different LLM providers.

**Features:**
- Provider-agnostic model interface
- Environment variable-based API key management
- Support for OpenAI, Groq, and Anthropic providers
- MCP server integration for tool execution

**Key Functions:**
- `get_model_provider()`: Returns appropriate model instance
- `run_model()`: Executes model with MCP server integration

#### Runner Utilities (`utilities/runner_utilities.py`)

Manages state persistence and workflow execution.

**Features:**
- Redis-based state caching
- State serialization/deserialization
- Task dependency resolution
- Conversation context preparation
- Cache management (load/save/clear)

**Key Functions:**
- `load_state()`: Loads agent state from cache
- `save_state()`: Persists agent state to cache
- `clear_cache()`: Clears cached state
- `find_matching_task()`: Resolves next task in workflow
- `prepare_conversation_context()`: Formats conversation history

#### UI Utilities (`utilities/ui_utilities.py`)

Generates UI components for different response types.

**Features:**
- Automatic response type detection
- Multiple UI component types (text, image, list)
- HTML formatting for rich content
- Responsive layout configuration
- LLM-based response classification

**Supported Response Types:**
1. **Conversational**: Simple chat responses
2. **Textual**: Rich text content with formatting
3. **Formatted List**: Structured list data
4. **Image**: Image URLs with display components

**Key Functions:**
- `generate_response_and_ui_components()`: Main UI generation entry point
- `generate_text_ui_component()`: Text display components
- `generate_image_ui_component()`: Image display components
- `generate_formatted_list_ui_component()`: List display components
- `generate_html_list_ui_component()`: HTML-formatted lists

## Workflow Execution

### 1. Configuration Loading
- Agent loads configuration from database or JSON file
- Parses enriched tasks into Task objects
- Validates dependencies and data flow

### 2. Workflow Building
- Creates LangGraph StateGraph from Task objects
- Adds nodes for each task (reasoning or tool-based)
- Establishes edges based on task dependencies
- Compiles workflow for execution

### 3. State Management
- Initializes data objects from task variables
- Loads cached state if available
- Prepares conversation context
- Manages execution history

### 4. Task Execution
- Finds next task based on current state
- Executes task using appropriate node type
- Updates data objects with results
- Determines next task in workflow

### 5. Response Generation
- Processes final workflow state
- Generates appropriate UI components
- Formats chat response
- Saves state for continuation

## Configuration Format

The system expects configuration in the following format:

```json
{
  "enriched_tasks": [
    {
      "id": "task_1",
      "input_dependencies": [],
      "output_dependencies": ["task_2"],
      "task_name": "Initial Analysis",
      "task_description": "Analyze user query",
      "tool_id": "",
      "tool_action": "",
      "tool_parameters": {},
      "model_id": "openai:gpt-4",
      "model_reason": "reasoning",
      "input_variables": [
        {"name": "user_query", "type": "string"}
      ],
      "output_variables": [
        {"name": "analysis", "type": "string"}
      ],
      "prompt_system": "You are a helpful assistant.",
      "prompt_user": "Analyze: {user_query}",
      "prompt_variables": []
    }
  ],
  "render_ui": true
}
```

## Environment Variables

The system requires the following environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `GROQ_API_KEY`: Groq API key  
- `ANTHROPIC_API_KEY`: Anthropic API key
- Redis configuration for state caching

## Usage Example

```python
from agent_runner.configurable_agent_executor import ConfigurableAgentExecutor

# Initialize agent
agent = ConfigurableAgentExecutor(name="workflow_agent", agent_id=1)

# Run workflow
result = await agent.run(
    query="Analyze this data",
    user_id=123,
    conversation_history=[...]
)

# Access results
print(result.response)
print(result.ui_components)
```

## Error Handling

The system includes comprehensive error handling:
- Configuration validation
- State persistence errors
- Model provider failures
- Tool execution errors
- UI generation failures
- Graceful fallbacks for all error types

## Dependencies

- LangGraph: Workflow orchestration
- SQLAlchemy: Database integration
- Redis: State caching
- OpenAI/Groq/Anthropic: LLM providers
- Pydantic-AI: MCP server integration
- FastAPI: Web framework integration 
