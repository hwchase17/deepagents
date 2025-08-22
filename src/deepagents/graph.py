from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import (
    write_todos,
    write_file,
    read_file,
    ls,
    edit_file,
    glob,
    grep,
)
from deepagents.local_fs_tools import (
    write_file as local_write_file,
    read_file as local_read_file,
    ls as local_ls,
    edit_file as local_edit_file,
    glob as local_glob,
    grep as local_grep,
    str_replace_based_edit_tool,
)
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.types import Checkpointer

from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage.

"""


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    local_filesystem: bool = False,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then six file system tools: write_file, ls, read_file, edit_file, glob, grep.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        local_filesystem: If True, use real filesystem tools instead of mock state-based tools
        config_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        post_model_hook: Optional post model hook function for intercepting tool calls.
    """
    prompt = instructions + base_prompt
    if local_filesystem:
        built_in_tools = [
            write_todos,
            local_write_file,
            local_read_file,
            local_ls,
            local_edit_file,
            local_glob,
            local_grep,
            str_replace_based_edit_tool,
        ]
    else:
        built_in_tools = [write_todos, write_file, read_file, ls, edit_file, glob, grep]
    
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(
        list(tools) + built_in_tools, instructions, subagents or [], model, state_schema
    )
    all_tools = built_in_tools + list(tools) + [task_tool]
    
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        config_schema=config_schema,
        checkpointer=checkpointer,
        post_model_hook=post_model_hook,
    )
