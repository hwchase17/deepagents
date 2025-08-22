"""
Post model hook for the coding agent with intelligent caching for command approvals.
Similar to Claude Code's approval system - only prompts for new commands or new directories.
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
import os


def create_coding_agent_post_model_hook():
    """Create a post model hook with intelligent approval caching."""
    
    def get_approval_key(command: str, args: Dict[str, Any]) -> str:
        """Generate a cache key for command approval based on command type and target directory."""
        # Extract directory from common file operations
        target_dir = None
        
        if command in ["write_file", "str_replace_based_edit_tool", "edit_file"]:
            file_path = args.get("file_path") or args.get("path")
            if file_path:
                target_dir = os.path.dirname(os.path.abspath(file_path))
        elif command == "execute_bash":
            # For bash commands, use current working directory or cwd from args
            target_dir = args.get("cwd") or os.getcwd()
        elif command in ["ls", "glob", "grep"]:
            # For read operations, use the path if provided
            target_dir = args.get("path") or args.get("directory") or os.getcwd()
        
        if not target_dir:
            target_dir = os.getcwd()
            
        # Create a cache key: command_type:normalized_directory
        normalized_dir = os.path.normpath(target_dir)
        return f"{command}:{normalized_dir}"
    
    def is_operation_approved(approved_operations: Dict[str, Any], command: str, args: Dict[str, Any]) -> bool:
        """Check if a command/directory combination has been previously approved."""
        if not approved_operations:
            return False
            
        approval_key = get_approval_key(command, args)
        return approval_key in approved_operations.get("cached_approvals", set())
    
    def add_approved_operation(approved_operations: Dict[str, Any], command: str, args: Dict[str, Any]):
        """Add a command/directory combination to the approved operations cache."""
        if "cached_approvals" not in approved_operations:
            approved_operations["cached_approvals"] = set()
            
        approval_key = get_approval_key(command, args)
        approved_operations["cached_approvals"].add(approval_key)
    
    def post_model_hook(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post model hook that checks for write tool calls and uses caching to avoid
        redundant approval prompts for the same command/directory combinations.
        """
        # Get the last message from the state
        messages = state.get("messages", [])
        if not messages:
            return state
        
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state
        
        # Get or initialize the approval cache from state
        approved_operations = state.get("approved_operations", {"cached_approvals": set()})
        
        approved_tool_calls = []
        
        # Define write tools that need approval
        write_tools = {
            "write_file", "execute_bash", "str_replace_based_edit_tool", "ls", "edit_file", "glob", "grep" 
        }
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            
            if tool_name in write_tools:
                # Check if this command/directory combination has been approved before
                if is_operation_approved(approved_operations, tool_name, tool_args):
                    # Already approved - execute without prompting
                    approved_tool_calls.append(tool_call)
                else:
                    # New command or directory - ask for approval
                    approval_key = get_approval_key(tool_name, tool_args)
                    question = (
                        f"New operation detected:\n\n"
                        f"Command: {tool_name}\n"
                        f"Directory: {approval_key.split(':', 1)[1]}\n"
                        f"Args: {tool_args}\n\n"
                        f"Approve this command for this directory? "
                        f"(Future identical operations in this directory will be auto-approved)\n\n"
                        f"Respond with True to approve or False to reject."
                    )
                    
                    is_approved = interrupt({
                        "question": question,
                        "command": tool_name,
                        "args": tool_args,
                        "approval_key": approval_key
                    })
                    
                    if is_approved:
                        # Add to cache and approve the tool call
                        add_approved_operation(approved_operations, tool_name, tool_args)
                        approved_tool_calls.append(tool_call)
                        
                        # Update the main state with the new approval cache
                        state["approved_operations"] = approved_operations
                    else:
                        # Rejected - skip this tool call
                        continue
            else:
                # For all other tools (read-only operations), allow them to execute without approval
                approved_tool_calls.append(tool_call)
        
        # Update the message if any tool calls were filtered out
        if len(approved_tool_calls) != len(last_message.tool_calls):
            new_message = AIMessage(
                content=last_message.content,
                tool_calls=approved_tool_calls,
                additional_kwargs=last_message.additional_kwargs
            )
            
            new_messages = messages[:-1] + [new_message]
            state["messages"] = new_messages
        
        return state
    
    return post_model_hook
