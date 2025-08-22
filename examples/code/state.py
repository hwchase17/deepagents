from deepagents.state import DeepAgentState
from typing import Dict, Set, NotRequired
from typing_extensions import TypedDict
import os
import hashlib

class CodingAgentState(DeepAgentState):
    """Extended state for coding agent with approval operation caching."""
    approved_operations: NotRequired[Dict[str, Set[str]]]
    
    def get_approval_key(self, command: str, args: Dict) -> str:
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
    
    def is_operation_approved(self, command: str, args: Dict) -> bool:
        """Check if a command/directory combination has been previously approved."""
        if not hasattr(self, 'approved_operations') or not self.approved_operations:
            return False
            
        approval_key = self.get_approval_key(command, args)
        return approval_key in self.approved_operations.get("cached_approvals", set())
    
    def add_approved_operation(self, command: str, args: Dict):
        """Add a command/directory combination to the approved operations cache."""
        if not hasattr(self, 'approved_operations') or self.approved_operations is None:
            self.approved_operations = {"cached_approvals": set()}
        
        if "cached_approvals" not in self.approved_operations:
            self.approved_operations["cached_approvals"] = set()
            
        approval_key = self.get_approval_key(command, args)
        self.approved_operations["cached_approvals"].add(approval_key)
