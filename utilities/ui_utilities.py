from typing import Dict, Any, Tuple, List
import logging
from src.services.llm.openai import OpenAIService
import json
#import html

logger = logging.getLogger(__name__)

""" 
This module contains functions for generating UI components from the agent's response.

It current works to identify four types of responses:
1) Conversational: The agent's response is a conversational response to the user's input. No UI components are needed.
2) Textual: The agent's response is the textual result of work completed and create markup of the agent's response. You will then need to select the appropriate UI component to use and format a json object with the data and the UI component. Then you need to generate an appropriate response to the user's input based upon the UI component and the data and set that as the chat response.  Call this option "textual".
3) Formatted List: The response is a list of one or more items that can be formatted for better readability
4) Image: The agent's response contains an image which is the result of work completed and format an image url to be displayed in the UI component. You will then need to select the appropriate UI component to use and format a json object with the image url and the UI component. Then you need to generate an appropriate response to the user's input based upon the UI component and the data and set that as the chat response. Call this option "image".

"""

def convert_to_basic_text(content: Dict[str, Any]) -> str:
    """Convert content dictionary to basic text format by iterating through all items."""
    try:
        if not content:
            return "No content available"
            
        if not isinstance(content, dict):
            logger.warning(f"Content is not a dictionary: {type(content)}")
            return str(content)
            
        # Build the basic text representation by iterating through all items
        basic_text = ""
        for key, value in content.items():
            try:
                if value is not None:  # Skip None values
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        nested_text = convert_to_basic_text(value)
                        if nested_text:
                            basic_text += f"{key}:\n{nested_text}\n"
                    elif isinstance(value, list):
                        # Handle lists
                        list_items = [str(item) for item in value if item is not None]
                        if list_items:
                            basic_text += f"{key}:\n" + "\n".join(f"- {item}" for item in list_items) + "\n"
                    else:
                        # Handle simple values
                        basic_text += f"{key}: {value}\n"
            except Exception as e:
                logger.warning(f"Error processing key '{key}': {str(e)}")
                basic_text += f"{key}: [Error processing value]\n"
                    
        return basic_text.strip()
    except Exception as e:
        logger.error(f"Error converting content to basic text: {str(e)}")
        return "Error processing content"


def generate_text_ui_component(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a UI component for displaying text content using a CardWithList component.
    
    Args:
        content: The text content to display
        
    Returns:
        Dict containing the UI component configuration
    """
    try:
        def format_text_for_html(text: str) -> str:
            """Format text for HTML display in a div by handling special characters that may cause display problems."""
            try:
                # First convert the text to a string and handle None
                if text is None:
                    return ""
                    
                formatted = str(text)
                
                # Handle special characters that may cause display problems in a div
                replacements = {
                    '&': ' ',
                    '<': ' ',
                    '>': ' ',
                    '"': ' ',
                    "'": ' '
                }
                
                for char, replacement in replacements.items():
                    formatted = formatted.replace(char, replacement)
                    
                return formatted
            except Exception as e:
                logger.warning(f"Error formatting text for HTML: {str(e)}")
                return str(text) if text is not None else ""

        formatted_content = format_text_for_html(content)

        return {
            "type": "GenericTextCard",
            "props": {
                "title": "Response",
                "description": "Generated content",
                "text": formatted_content,
                "layoutConfig": {
                    "xs": {"chatWidth": "100%", "contentWidth": "100%"},
                    "sm": {"chatWidth": "100%", "contentWidth": "100%"},
                    "md": {"chatWidth": "65%", "contentWidth": "35%"},
                    "lg": {"chatWidth": "65%", "contentWidth": "35%"},
                    "xl": {"chatWidth": "65%", "contentWidth": "35%"},
                    "2xl": {"chatWidth": "65%", "contentWidth": "35%"}
                }
            },
            "position": "right"
        }
    except Exception as e:
        logger.error(f"Error generating text UI component: {str(e)}")
        return {}

def generate_image_ui_component(image_url: str, alt_text: str = "Generated image") -> Dict[str, Any]:
    """
    Generate a UI component for displaying an image using a Card component.
    
    Args:
        image_url: The URL of the image to display
        alt_text: Alternative text for the image
        
    Returns:
        Dict containing the UI component configuration
    """
    try:
        if not image_url:
            logger.warning("No image URL provided")
            return {}

        return {
            "type": "GenericImageCard",
            "props": {
                "title": "Generated Image",
                "description": alt_text,
                "url": image_url,
                "layoutConfig": {
                    "xs": {"chatWidth": "100%", "contentWidth": "100%"},
                    "sm": {"chatWidth": "100%", "contentWidth": "100%"},
                    "md": {"chatWidth": "65%", "contentWidth": "35%"},
                    "lg": {"chatWidth": "65%", "contentWidth": "35%"},
                    "xl": {"chatWidth": "65%", "contentWidth": "35%"},
                    "2xl": {"chatWidth": "65%", "contentWidth": "35%"},
                },
            },
            "position": "right"
        }
    except Exception as e:
        logger.error(f"Error generating image UI component: {str(e)}")
        return {}

def format_list_items(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format content into a list of items with highlights and details.
    
    Args:
        content: Dictionary containing the data to format
        
    Returns:
        List of formatted items with highlight and details
    """
    try:
        if not content:
            return []

        def is_url(text: str) -> bool:
            """Check if a string is a URL."""
            return text.startswith(('http://', 'https://', 'www.'))

        def format_value(value: Any) -> str:
            """Format a value, converting URLs to links."""
            if isinstance(value, str) and is_url(value):
                return f'<a href="{value}" target="_blank">{value}</a>'
            return str(value)

        # Process the content into formatted items
        formatted_items = []
        for key, value in content.items():
            if value is not None:
                if isinstance(value, list):
                    # Process list items
                    for item in value:
                        if isinstance(item, dict):
                            # Get the first value for highlighting
                            first_key = next(iter(item.keys()), None)
                            if first_key:
                                formatted_item = {
                                    "highlight": str(item[first_key]),
                                    "details": []
                                }
                                # Add remaining values
                                for k, v in item.items():
                                    if k != first_key:
                                        formatted_item["details"].append({
                                            "label": k,
                                            "value": format_value(v)
                                        })
                                formatted_items.append(formatted_item)
                        else:
                            # Handle simple list items
                            formatted_items.append({
                                "highlight": str(item),
                                "details": []
                            })
                elif isinstance(value, dict):
                    # Process dictionary items
                    first_key = next(iter(value.keys()), None)
                    if first_key:
                        formatted_item = {
                            "highlight": str(value[first_key]),
                            "details": []
                        }
                        for k, v in value.items():
                            if k != first_key:
                                formatted_item["details"].append({
                                    "label": k,
                                    "value": format_value(v)
                                })
                        formatted_items.append(formatted_item)
                else:
                    # Handle simple values
                    formatted_items.append({
                        "highlight": str(value),
                        "details": []
                    })

        return formatted_items
    except Exception as e:
        logger.error(f"Error formatting list items: {str(e)}")
        return []

def generate_formatted_list_ui_component(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a UI component for displaying formatted list content.
    
    Args:
        content: Dictionary containing list data to format
        
    Returns:
        Dict containing the UI component configuration
    """
    try:
        formatted_items = format_list_items(content)
        
        if not formatted_items:
            return {
                "type": "GenericListCard",
                "props": {
                    "title": "No Content",
                    "description": "No items available",
                    "items": [],
                    "layoutConfig": {
                        "xs": {"chatWidth": "100%", "contentWidth": "100%"},
                        "sm": {"chatWidth": "100%", "contentWidth": "100%"},
                        "md": {"chatWidth": "65%", "contentWidth": "35%"},
                        "lg": {"chatWidth": "65%", "contentWidth": "35%"},
                        "xl": {"chatWidth": "65%", "contentWidth": "35%"},
                        "2xl": {"chatWidth": "65%", "contentWidth": "35%"}
                    }
                },
                "position": "right"
            }

        return {
            "type": "GenericListCard",
            "props": {
                "title": "Results",
                "description": f"{len(formatted_items)} item(s) found",
                "items": formatted_items,
                "layoutConfig": {
                    "xs": {"chatWidth": "100%", "contentWidth": "100%"},
                    "sm": {"chatWidth": "100%", "contentWidth": "100%"},
                    "md": {"chatWidth": "65%", "contentWidth": "35%"},
                    "lg": {"chatWidth": "65%", "contentWidth": "35%"},
                    "xl": {"chatWidth": "65%", "contentWidth": "35%"},
                    "2xl": {"chatWidth": "65%", "contentWidth": "35%"}
                }
            },
            "position": "right"
        }
    except Exception as e:
        logger.error(f"Error generating formatted list UI component: {str(e)}")
        return {}

def generate_html_list_ui_component(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an HTML UI component for displaying formatted list content.
    
    Args:
        content: Dictionary containing the data to format
        
    Returns:
        Dict containing the UI component configuration
    """
    try:
        formatted_items = format_list_items(content)
        
        if not formatted_items:
            return {
                "type": "GenericTextCard",
                "props": {
                    "title": "No Content",
                    "description": "No items available",
                    "text": "<p>No items available</p>",
                    "layoutConfig": {
                        "xs": {"chatWidth": "100%", "contentWidth": "100%"},
                        "sm": {"chatWidth": "100%", "contentWidth": "100%"},
                        "md": {"chatWidth": "65%", "contentWidth": "35%"},
                        "lg": {"chatWidth": "65%", "contentWidth": "35%"},
                        "xl": {"chatWidth": "65%", "contentWidth": "35%"},
                        "2xl": {"chatWidth": "65%", "contentWidth": "35%"}
                    }
                },
                "position": "right"
            }

        html_parts = ['<div class="formatted-list">']
        
        for item in formatted_items:
            html_parts.append('<div class="formatted-item">')
            
            # Add highlight
            if "highlight" in item:
                html_parts.append(f'<div class="highlight">{item["highlight"]}</div>')
            
            # Add details
            if "details" in item and item["details"]:
                html_parts.append('<div class="details">')
                for detail in item["details"]:
                    if "label" in detail and "value" in detail:
                        html_parts.append(f'<div class="detail-item">')
                        html_parts.append(f'<span class="label">{detail["label"]}:</span>')
                        html_parts.append(f'<span class="value">{detail["value"]}</span>')
                        html_parts.append('</div>')
                html_parts.append('</div>')
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        # Add some basic CSS
        css = """
        <style>
            .formatted-list {
                padding: 1rem;
            }
            .formatted-item {
                margin-bottom: 1rem;
                padding: 1rem;
                border: 1px solid #e2e8f0;
                border-radius: 0.5rem;
            }
            .highlight {
                font-weight: bold;
                font-size: 1.1em;
                margin-bottom: 0.5rem;
                color: #2d3748;
            }
            .details {
                margin-left: 1rem;
            }
            .detail-item {
                margin: 0.25rem 0;
            }
            .label {
                font-weight: 500;
                color: #4a5568;
                margin-right: 0.5rem;
            }
            .value {
                color: #2d3748;
            }
            .value a {
                color: #3182ce;
                text-decoration: none;
            }
            .value a:hover {
                text-decoration: underline;
            }
        </style>
        """
        
        html_content = css + '\n'.join(html_parts)
        
        return {
            "type": "GenericTextCard",
            "props": {
                "title": "Results",
                "description": f"{len(formatted_items)} item(s) found",
                "text": html_content,
                "layoutConfig": {
                    "xs": {"chatWidth": "100%", "contentWidth": "100%"},
                    "sm": {"chatWidth": "100%", "contentWidth": "100%"},
                    "md": {"chatWidth": "65%", "contentWidth": "35%"},
                    "lg": {"chatWidth": "65%", "contentWidth": "35%"},
                    "xl": {"chatWidth": "65%", "contentWidth": "35%"},
                    "2xl": {"chatWidth": "65%", "contentWidth": "35%"}
                }
            },
            "position": "right"
        }
        
    except Exception as e:
        logger.error(f"Error generating HTML list UI component: {str(e)}")
        return {}

async def generate_response_and_ui_components(response: Any, user_query: str) -> Tuple[str, list[Dict[str, Any]]]:
    """
    Process a response and generate appropriate UI components.
    
    Args:
        response: The response data which can be a string, dictionary, or other format
        
    Returns:
        Tuple containing:
        - str: The processed response text
        - list[Dict[str, Any]]: List of UI component configurations
    """
    try:
        # prep the llm
        system_prompt = """You are a helpful assistant that generates the appropriate response to a user, based upon the response an AI agent has created from the user's input.  You are receiving the response from the AI agent and the user's input. There are two things you need to determine. First you need to determine if the response from the agent should be displayed in a separate UI panel or if it should simply be returned as part of the chat response. If a separate UI panel is appropriate the second thing you need to determine is which UI component to use.  Chose one of the four following options that best fits given the response from the AI agent: 

        1) If the AI agent's response is simply a conversational response to a user's query or a question for the user, set the chat response to the agent's response. No UI components are needed. Call this option "conversational".
        2) If the AI agent's response provided information beyond a simple conversational response, then a UI panel is appropriate.  You will need to select the appropriate UI component to use and format a json object with the agent response in it to send to the UI component. You also need to generate an appropriate chat response to the user in the voice of the agent. This should signifying the agent had responded.  Call this option "textual".
        3) If the AI agent response is a list of one or more items, then a UI panel in appropriate.  You will need to select the appropriate UI component to use and format a json object with the data to send to the UI component. You also need to generate an appropriate chat response to the user in the voice of the agent. This should signifying the agent had responded. Call this option "formatted_list".
        4) If the AI agent response contains a url to an image, then a UI panel is appropriate. Verigy the URL links to an actual image. You will need to select the appropriate UI component to use and format a json object with the image url to send to the UI component. You will need to select the appropriate UI component to use and format a json object with the data to send to the UI component.  You also need to generate an appropriate chat response to the user in the voice of the agent. This should signifying the agent had responded. Call this option "image".
        
        You need to generate and return the chat response and data to use in the UI components as a json object.  The json object will have the following format:
        {
            "chat_response": "The chat response to the user's input",
            "ui_data": "The UI data to be displayed to the user"
            "option": "The option you selected"
        }"""

        user_prompt = f"""
        The user's input is: {user_query}
        The agent's response is: {response}
        """

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
        openai_service = OpenAIService(
                model_name="gpt-4o-mini",
                temperature=0.2,
                max_tokens=1024
            )
        draft_llmresponse = await openai_service.chat_completion(messages)

        if draft_llmresponse and draft_llmresponse.choices and len(draft_llmresponse.choices) > 0:
            llm_response = draft_llmresponse.choices[0].message.content
            logger.debug(f"Draft LLM response: {llm_response}")
            try:
                # Clean the response string by removing markdown code block markers
                llm_response = llm_response.strip()
                if llm_response.startswith('```'):
                    llm_response = llm_response.split('\n', 1)[1]  # Remove first line with ```
                if llm_response.endswith('```'):
                    llm_response = llm_response.rsplit('\n', 1)[0]  # Remove last line with ```
                llm_response = json.loads(llm_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return str(response), []

        if isinstance(llm_response, dict) and "option" in llm_response:
            if llm_response["option"] == "conversational":
                chat_response = llm_response["chat_response"]
                ui_components = []
            elif llm_response["option"] == "textual":
                chat_response = llm_response["chat_response"]
                ui_components = [generate_text_ui_component(llm_response["ui_data"])]
            elif llm_response["option"] == "formatted_list":
                chat_response = llm_response["chat_response"]
                ui_components = [generate_formatted_list_ui_component(llm_response["ui_data"])]
            elif llm_response["option"] == "image":
                chat_response = llm_response["chat_response"]
                ui_components = [generate_image_ui_component(llm_response["ui_data"]["url"])]
            else:
                chat_response = response
                ui_components = []
        
        return chat_response, ui_components
        
    except Exception as e:
        logger.error(f"Error generating response and UI components: {str(e)}")
        # Return a safe fallback
        default_response = response
        return default_response, []