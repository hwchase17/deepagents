import os
import subprocess
import tempfile
import platform
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from deepagents import create_deep_agent, SubAgent


def execute_bash(command: str, timeout: int = 30, cwd: str = None) -> Dict[str, Any]:
    """
    Execute bash/shell commands safely.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        cwd: Working directory for command execution

    Returns:
        Dictionary with execution results including stdout, stderr, and success status
    """
    try:
        # Determine the appropriate shell based on platform
        if platform.system() == "Windows":
            shell_cmd = ["cmd", "/c", command]
        else:
            shell_cmd = ["bash", "-c", command]

        # Execute the command
        result = subprocess.run(
            shell_cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "return_code": -1,
        }


def http_request(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    data: Union[str, Dict] = None,
    params: Dict[str, str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        # Prepare request parameters
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        # Make the request
        response = requests.request(**kwargs)

        # Try to parse JSON response, fallback to text
        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {str(e)}",
            "url": url,
        }


# Sub-agent for code review and analysis
code_reviewer_prompt = """You are an expert code reviewer for all programming languages. Your job is to analyze code for:

1. **Code Quality**: Check for clean, readable, and maintainable code
2. **Best Practices**: Ensure adherence to language-specific best practices and conventions
3. **Security**: Identify potential security vulnerabilities
4. **Performance**: Suggest optimizations where applicable
5. **Testing**: Evaluate test coverage and quality
6. **Documentation**: Check for proper comments and documentation

When reviewing code, provide:
- Specific line-by-line feedback
- Language-specific suggestions for improvements
- Security concerns (if any)
- Performance optimization opportunities
- Overall assessment and rating (1-10)

You can use bash commands to run linters, formatters, and other code analysis tools for any language.
Be constructive and educational in your feedback. Focus on helping improve the code quality."""

code_reviewer_agent = {
    "name": "code-reviewer",
    "description": "Expert code reviewer that analyzes code in any programming language for quality, security, performance, and best practices. Use this when you need detailed code analysis and improvement suggestions.",
    "prompt": code_reviewer_prompt,
    "tools": ["execute_bash"],
}

# Sub-agent for debugging assistance
debugger_prompt = """You are an expert debugging assistant for all programming languages. Your job is to help identify and fix bugs in any codebase.

When debugging:
1. **Analyze Error Messages**: Interpret stack traces and error messages across languages
2. **Identify Root Causes**: Find the underlying cause of issues
3. **Suggest Fixes**: Provide specific solutions and code corrections
4. **Test Solutions**: Verify that fixes work correctly using appropriate tools
5. **Explain Issues**: Help understand why the bug occurred

You have access to bash commands to run compilers, interpreters, debuggers, and other development tools.
Always validate your solutions by testing the corrected code with the appropriate language tools.

Be systematic in your approach:
- First understand the error
- Identify the problematic code section
- Propose a fix
- Test the fix using language-specific tools
- Explain the solution"""

debugger_agent = {
    "name": "debugger",
    "description": "Expert debugging assistant that helps identify and fix bugs in any programming language. Use when you encounter errors or unexpected behavior in code.",
    "prompt": debugger_prompt,
    "tools": ["execute_bash"],
}

# Sub-agent for test generation
test_generator_prompt = """You are an expert test engineer for all programming languages. Your job is to create comprehensive test suites for any codebase.

When generating tests:
1. **Test Coverage**: Create tests that cover all functions, methods, and edge cases
2. **Test Types**: Include unit tests, integration tests, and edge case tests
3. **Frameworks**: Use appropriate testing frameworks for each language (Jest, pytest, JUnit, Go test, etc.)
4. **Assertions**: Write meaningful assertions that validate expected behavior
5. **Documentation**: Include clear test descriptions and comments

Test categories to consider:
- **Happy Path**: Normal expected inputs and outputs
- **Edge Cases**: Boundary conditions, empty inputs, large inputs
- **Error Cases**: Invalid inputs, exception handling
- **Integration**: How components work together

Use bash commands to run language-specific test frameworks and verify that tests execute successfully.
Always verify that your tests can run successfully and provide meaningful feedback."""

test_generator_agent = {
    "name": "test-generator",
    "description": "Expert test engineer that creates comprehensive test suites for any programming language. Use when you need to generate thorough test suites for your code.",
    "prompt": test_generator_prompt,
    "tools": ["execute_bash"],
}

# Main coding agent instructions
coding_instructions = """You are an expert software developer and coding assistant. Your job is to help users with all aspects of programming across multiple languages including:

## Core Capabilities
- **Code Development**: Write clean, efficient, and well-documented code in any language
- **Problem Solving**: Break down complex problems into manageable solutions
- **Code Review**: Analyze and improve existing code across languages
- **Debugging**: Identify and fix bugs in any programming language
- **Testing**: Create comprehensive test suites using appropriate frameworks
- **Code Optimization**: Improve performance and efficiency
- **System Operations**: Handle build tools, package managers, and development environments

## Workflow
1. **Understand Requirements**: Carefully analyze what the user needs
2. **Plan Solution**: Break down the problem into steps
3. **Implement Code**: Write clean, well-documented code
4. **Test Code**: Verify functionality with comprehensive tests
5. **Review & Optimize**: Use sub-agents for code review and improvements

## Available Sub-Agents
- **code-reviewer**: For detailed code analysis and quality assessment across languages
- **debugger**: For identifying and fixing bugs in any programming language
- **test-generator**: For creating comprehensive test suites using appropriate frameworks

## Best Practices
- Write appropriate documentation for functions and classes
- Follow language-specific style guidelines and conventions
- Handle errors gracefully with appropriate exception handling
- Write meaningful variable and function names
- Use language-appropriate type systems when available
- Create comprehensive tests for your code

## Tools Available
- **execute_bash**: Run shell commands for compilation, testing, package management, etc.
- **http_request**: Make API calls, download resources, interact with web services

## File Management
- Save code to appropriate files with correct extensions
- Organize projects with proper directory structure
- Create documentation files when needed (README.md, etc.)
- Use version control best practices

## Development Operations
You can handle:
- Building and compiling code (make, cmake, cargo build, etc.)
- Package management (npm, pip, gem, cargo, etc.)
- Running tests (pytest, jest, junit, go test, etc.)
- Code formatting (prettier, black, gofmt, etc.)
- Static analysis and linting
- Environment setup and configuration

Always test your code using appropriate tools before presenting it to the user. If there are errors, use the debugger sub-agent to help identify and fix issues.

Remember: Quality code is more important than quick code. Take time to write clean, tested, and well-documented solutions."""

# Create the coding agent
agent = create_deep_agent(
    [execute_bash, http_request],
    coding_instructions,
    subagents=[code_reviewer_agent, debugger_agent, test_generator_agent],
    local_filesystem=True,
).with_config({"recursion_limit": 1000})
