# Jerzy: A Framework for Explainable LLM-Powered Agents

Jerzy is a lightweight, modular Python framework for building explainable and auditable AI agents powered by large language models (LLMs). It provides transparent reasoning, comprehensive tracing, and detailed audit trails while simplifying the development of LLM applications.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
  - [Creating a Simple Agent](#creating-a-simple-agent)
  - [Adding Tools](#adding-tools)
  - [Running the Agent](#running-the-agent)
- [Advanced Features](#advanced-features)
  - [Memory and Conversation Management](#memory-and-conversation-management)
  - [Tracing and Auditing](#tracing-and-auditing)
  - [Planning Capabilities](#planning-capabilities)
  - [Multi-Agent Systems](#multi-agent-systems)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Key Features

- **Transparent Reasoning**: Capture and expose the reasoning processes of LLMs
- **Tool Integration**: Seamlessly connect LLMs with external tools and APIs
- **Comprehensive Auditing**: Track token usage, prompts, completions, and tool calls
- **Conversation Memory**: Maintain context across multiple interactions
- **Execution Tracing**: Detailed logs of execution steps for better explainability
- **Planning Capabilities**: Generate and execute structured plans to achieve complex goals
- **Multi-Agent Collaboration**: Coordinate multiple specialized agents to solve problems
- **Robust Tool Calling**: Support for both native function calling and text-based parsing

## Installation

```bash
pip install git+https://github.com/JerzyKultura/Jerzy.git
```

## Core Concepts

Jerzy is built around several core components:

- **Agent**: The main interface for interacting with LLMs, managing tools, memory, and executing queries
- **LLM**: Abstraction for language model providers (currently supports OpenAI-compatible APIs)
- **Tool**: Wrapper around functions that LLMs can call to perform actions
- **Memory**: Systems for storing conversation history and context
- **Trace**: Components for logging execution steps and producing audit trails
- **State**: Manages evolving knowledge and configuration during agent execution
- **Chain**: Composes multiple operations into workflows
- **Plan/Planner**: Creates and executes structured plans for complex goals

## Basic Usage

### Creating a Simple Agent

```python
from jerzy import OpenAILLM, Agent

# Initialize the LLM
llm = OpenAILLM(api_key="your-api-key", model="gpt-4")

# Create an agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant who excels at solving problems.",
    enable_auditing=True  # Enable detailed audit trails
)

# Run a basic query
response, history = agent.run(
    user_query="What is the capital of France?",
    verbose=True  # Print execution details
)

print(response)
```

### Adding Tools

```python
from jerzy import Tool

# Define a simple tool
def search_web(query: str) -> dict:
    """Search the web for information."""
    # Implementation would connect to a search API
    return {"results": [f"Result for {query}", "Another result"]}

# Create a Tool from the function
search_tool = Tool(
    name="search_web",
    func=search_web,
    description="Search the web for current information on a topic",
    cacheable=True  # Results can be cached
)

# Add the tool to the agent
agent.add_tools([search_tool])

# Now the agent can use this tool
response, history = agent.run(
    user_query="What are the latest developments in AI?",
    max_steps=3  # Allow up to 3 tool calls/steps
)
```

### Running the Agent

```python
# Configure reasoning verbosity
response, history = agent.run(
    user_query="Analyze the current economic situation in Europe",
    max_steps=5,
    verbose=True,
    reasoning_mode="full",  # Options: "none", "short", "medium", "full"
    use_cache=True,
    allow_repeated_calls=False
)

# Get structured output with traces
result = agent.run(
    user_query="Analyze stock market trends for tech companies",
    return_trace=True
)

print(f"Response: {result['response']}")
print(f"Trace: {result['trace']}")
```

## Advanced Features

### Memory and Conversation Management

Jerzy provides rich conversation memory with threading support:

```python
from jerzy import ConversationalAgent

# Create a conversational agent
conv_agent = ConversationalAgent(
    llm=llm,
    system_prompt="You are a helpful assistant who remembers past conversations.",
    use_vector_memory=False  # Set to True for semantic search capabilities
)

# Start a conversation
thread_id = conv_agent.start_new_conversation()

# Have multiple exchanges in the same thread
response1 = conv_agent.chat("Tell me about quantum computing", thread_id=thread_id)
response2 = conv_agent.chat("What were the key points you mentioned?", thread_id=thread_id)

# Get conversation history
history = conv_agent.get_conversation_history(thread_id, formatted=True)
print(history)

# List all conversations
conversations = conv_agent.list_conversations()
```

### Tracing and Auditing

Jerzy provides detailed tracing and auditing capabilities:

```python
# Run an agent with auditing enabled
agent = Agent(llm, enable_auditing=True)
response, _ = agent.run("Analyze this data set")

# Get audit summary
audit_summary = agent.get_audit_summary()
print(f"Total tokens used: {audit_summary['token_usage']['total_tokens']}")
print(f"Tool usage: {audit_summary['tool_usage']}")

# Save the audit trail to a file
audit_file = agent.save_audit_trail("/path/to/audit.json")

# Get a formatted trace
from jerzy import Trace
trace = Trace(agent.memory)
markdown_trace = trace.format_trace(format_type="markdown")
```

### Planning Capabilities

For complex tasks, Jerzy offers planning capabilities:

```python
from jerzy import EnhancedAgent

# Create an agent with planning abilities
planner_agent = EnhancedAgent(llm)

# Add tools the planner can use
planner_agent.add_tools([search_tool, another_tool])

# Define a complex goal
result = planner_agent.plan_and_execute(
    goal="Research the impact of climate change on agriculture in the last decade",
    context="Focus on major crop yields and adaptation strategies",
    verbose=True
)

print(result['summary'])
```

### Multi-Agent Systems

Jerzy supports coordinating multiple specialized agents:

```python
from jerzy import MultiAgentSystem, AgentRole

# Create roles for different agents
researcher_role = AgentRole(
    name="Researcher",
    description="Specializes in finding and analyzing information",
    system_prompt="You are a research specialist who excels at finding accurate information."
)

critic_role = AgentRole(
    name="Critic",
    description="Evaluates and critiques information for accuracy and bias",
    system_prompt="You carefully analyze claims for logical fallacies and factual errors."
)

# Create the multi-agent system
mas = MultiAgentSystem(llm)

# Add agents with their roles
mas.add_agent("researcher", researcher_role, [search_tool])
mas.add_agent("critic", critic_role)

# Have agents collaborate to solve a problem
solution = mas.collaborative_solve(
    problem="Evaluate the evidence for and against increasing nuclear power usage",
    max_turns=6,
    verbose=True
)

print(solution['summary'])

# Or have agents debate a topic
debate_result = mas.debate(
    topic="Is artificial general intelligence likely within the next decade?",
    rounds=3,
    verbose=True
)

print(debate_result['conclusion'])
```

## API Reference

### Agent Classes

- `Agent`: Core agent class with tool integration and reasoning
- `ConversationalAgent`: Specialized for multi-turn conversations
- `EnhancedAgent`: Agent with planning capabilities
- `MultiAgentSystem`: Coordinates multiple agents with different roles

### Tools and Function Calling

- `Tool`: Wrapper for functions that LLMs can use
- `ToolCache`: Caches tool results to reduce redundant calls
- `robust_tool`: Decorator for adding retry logic to tools

### Memory Systems

- `Memory`: Basic memory store
- `EnhancedMemory`: Advanced memory with conversation threading and retrieval

### Tracing and Auditing

- `Trace`: Captures execution traces
- `AuditTrail`: Detailed tracking of token usage, prompts, responses, and tool calls

### Planning

- `Plan`: Represents a structured plan with steps
- `Planner`: Creates and executes plans

## Examples

### Creating a Research Assistant

```python
import jerzy

# Create specialized research tools
def search_academic_papers(query: str, max_results: int = 5):
    """Search for academic papers on a topic."""
    # Implementation would connect to an academic API
    return {"papers": [f"Paper about {query}", "Another paper"]}

def analyze_sentiment(text: str):
    """Analyze the sentiment of a text."""
    # Implementation would use an NLP model
    return {"sentiment": "positive", "confidence": 0.85}

# Create tools
paper_tool = jerzy.Tool("search_papers", search_academic_papers,
                      "Search for academic papers on a topic")
sentiment_tool = jerzy.Tool("analyze_sentiment", analyze_sentiment,
                          "Analyze sentiment of text")

# Initialize LLM and agent
llm = jerzy.OpenAILLM(api_key="your-key", model="gpt-4")
research_agent = jerzy.Agent(llm, system_prompt="You are a research assistant.")
research_agent.add_tools([paper_tool, sentiment_tool])

# Run a research query
result = research_agent.run(
    "Research the impact of social media on teenage mental health",
    max_steps=10,
    reasoning_mode="full",
    return_trace=True
)

# Access the full execution trace
print(result["trace"])
```

## Best Practices

1. **Effective Tool Design**:
   - Keep tools focused on single functionalities
   - Provide clear descriptions and parameter specifications
   - Consider making stateless tools to improve caching

2. **Reasoning Modes**:
   - Use different reasoning modes based on needs:
     - "none": Fastest, but no transparency
     - "short": Quick summary of approach
     - "medium": Balanced detail and conciseness
     - "full": Complete transparency but higher token usage

3. **Memory Management**:
   - For long conversations, consider using `prune_history()` on memory objects
   - Use thread IDs to organize conversations by topic

4. **Audit Trails**:
   - Enable auditing for production systems to track usage and debug issues
   - Regularly review audit summaries for unexpected patterns

5. **Error Handling**:
   - Wrap tools in the `robust_tool` decorator for automatic retry logic
   - Use the `with_fallback` decorator to provide alternative implementations
