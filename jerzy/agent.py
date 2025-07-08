# agent.py - Central LLM-powered agent interface

from jerzy.common import *


from typing import List, Any, Optional
from .core import ToolCache, State, Tool
from .trace import Trace, AuditTrail, Plan, Planner
from .memory import Memory, EnhancedMemory
from .chain import Chain, ConversationChain
from .llm import LLM, OpenAILLM, CustomOpenAILLM


class Agent:
    """Enhanced LLM-powered agent with transparency, reasoning, and caching capabilities."""

    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100,
                 enable_auditing: bool = True):
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.tools = []
        self.memory = Memory()
        self.trace = Trace(self.memory)
        self.cache = ToolCache(max_size=cache_size, ttl=cache_ttl)
        self.state = State()
        self.tool_call_history = []

        # Initialize audit trail if enabled
        if enable_auditing:
            self.audit_trail = AuditTrail()
            # If LLM supports audit_trail, connect it
            if hasattr(self.llm, 'audit_trail'):
                self.llm.audit_trail = self.audit_trail
        else:
            self.audit_trail = None


    def _compare_args(self, args1: dict, args2: dict) -> bool:
        """Compare two argument dictionaries to detect duplicate tool calls."""
        try:
            # Simple comparison - convert to JSON strings and compare
            import json

            # Sort keys to ensure consistent comparison
            json1 = json.dumps(args1, sort_keys=True)
            json2 = json.dumps(args2, sort_keys=True)

            return json1 == json2

        except Exception:
            # Fallback to basic dict comparison if JSON serialization fails
            return args1 == args2

    def _extract_info_from_tool_result(self, tool_name: str, tool_args: dict, tool_result: dict) -> None:
        """Optionally extract structured insights from tool results for future use (e.g., memory, trace)."""

        if not hasattr(self, "memory") or not isinstance(tool_result, dict):
            return

    def add_tools(self, tools: List[Any]) -> None:
        """Register a list of tools with the agent, avoiding duplicates."""
        for tool in tools:
            if tool.name not in {t.name for t in self.tools}:
                self.tools.append(tool)

    def chat(self, user_message: str, thread_id: str = "default",
             use_semantic_search: bool = False, context_window: int = 10) -> str:
        """Have a conversation with the agent, with memory of past interactions."""
        # Initialize conversation chain if needed
        if not hasattr(self, 'conversation') or not self.conversation:
            self.conversation = ConversationChain(
                self.llm,
                EnhancedMemory(),
                self.system_prompt
            )

        # Generate response (with or without semantic search)
        if use_semantic_search:
            return self.conversation.search_and_respond(
                user_message, thread_id, context_window
            )
        else:
            return self.conversation.generate_response(
                user_message, thread_id, context_window
            )

    def remember(self, key: str, value: Any) -> None:
        """Explicitly store information in memory for later reference."""
        # Initialize conversation chain if needed
        if not hasattr(self, 'conversation') or not self.conversation:
            self.conversation = ConversationChain(
                self.llm,
                EnhancedMemory(),
                self.system_prompt
            )

        # Store in memory
        self.conversation.memory.set(key, value)

        # Add a system note about this
        self.conversation.add_message(
            "system",
            f"Stored information: {key} = {str(value)}",
            "default"
        )

    def save_conversation(self, filepath: str) -> None:
        """Save the current conversation to a file."""
        if hasattr(self, 'conversation') and self.conversation:
            self.conversation.save_conversation(filepath)
        else:
            print("No conversation to save.")

    def load_conversation(self, filepath: str) -> None:
        """Load a conversation from a file."""
        if not hasattr(self, 'conversation') or not self.conversation:
            self.conversation = ConversationChain(
                self.llm,
                EnhancedMemory(),
                self.system_prompt
            )

        self.conversation.load_conversation(filepath)

    def get_audit_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the audit trail if auditing is enabled."""
        if hasattr(self, 'audit_trail') and self.audit_trail:
            return self.audit_trail.get_summary()
        return None

    def save_audit_trail(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save the audit trail to a file if auditing is enabled."""
        if hasattr(self, 'audit_trail') and self.audit_trail:
            return self.audit_trail.save(filepath)
        return None

    def run(self, user_query: str, max_steps: int = 5, verbose: bool = False,
            return_trace: bool = False, use_cache: bool = True,
            reasoning_mode: str = "medium", allow_repeated_calls: bool = False) -> Union[tuple, Dict[str, Any]]:
        """Run the agent with configurable reasoning verbosity and auditing."""
        run_start_time = time.time()

        # Start new audit session for this run if auditing is enabled
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.start_session({
                "user_query": user_query,
                "max_steps": max_steps,
                "reasoning_mode": reasoning_mode,
                "use_cache": use_cache,
                "system_prompt": self.system_prompt
            })

        # Initialize state with query
        self.state.set("query.raw", user_query)
        self.state.set("query.timestamp", datetime.now().isoformat())
        self.state.set("execution.max_steps", max_steps)
        self.state.set("execution.reasoning_mode", reasoning_mode)
        self.state.set("execution.use_cache", use_cache)
        self.state.set("execution.allow_repeated_calls", allow_repeated_calls)

        # Reset tool call history for this run
        self.tool_call_history = []
        self.state.set("tools.called", {})
        self.state.set("tools.errors", {})

        # Define reasoning prompts with varying levels of detail
        reasoning_prompts = {
            "short": [
                {"role": "system", "content": "Summarize your approach to this query in 1-2 clear sentences."},
                {"role": "user", "content": f"Query: {user_query}\n\nSummarize approach:"}
            ],
            "medium": [
                {"role": "system", "content": "Explain your approach in 3-6 sentences, balancing clarity with detail."},
                {"role": "user", "content": f"Query: {user_query}\n\nOutline approach:"}
            ],
            "full": [
                {"role": "system", "content": "Explain your thinking step by step in detail."},
                {"role": "user",
                 "content": f"Query: {user_query}\n\nWhat tools would help answer this? How will you approach this problem?"}
            ]
        }

        # Use the cache only if explicitly requested
        active_cache = self.cache if use_cache else None

        # Initialize conversation with system prompt
        conversation = [{
            "role": "system",
            "content": self.system_prompt
        }]

        # Add cache guidance to system prompt if using cache
        if use_cache:
            cache_guidance = """
    You have access to cached results from previous tool calls.
    When you see a tool result marked as "cached", it means this data was retrieved from cache rather than calling the tool again.
    Consider whether cached data is still relevant for the current query before using it.
    If you need fresh data, explicitly state that you want to ignore the cache for a specific tool call by setting bypass_cache to true.
    """
            conversation[0]["content"] += "\n" + cache_guidance

        # Add guidance about repeated calls if duplicate detection is enabled
        if not allow_repeated_calls:
            repeat_guidance = """
    Be efficient with tool calls. If you've already called a tool with specific parameters, use that information
    rather than calling the same tool again with the same parameters.

    If you genuinely need to call the same tool again with the same parameters (e.g., for time-sensitive data
    or verification), add "force_repeat": true to the parameters to indicate this is intentional.
    """
            conversation[0]["content"] += "\n" + repeat_guidance

        # Add user query
        conversation.append({
            "role": "user",
            "content": user_query
        })

        # Record in memory
        self.memory.add_to_history({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })

        step = 0
        final_response = ""

        while step < max_steps:
            # Track step in state
            self.state.set("execution.current_step", step)

            # We will hit the max steps if the model keeps choosing to use tools
            if step == max_steps - 1:
                # For the last step, explicitly ask for a final answer
                conversation.append({
                    "role": "user",
                    "content": "Please provide your final answer based on the information above."
                })

            # First, get reasoning about how to approach the query
            if step == 0 and self.tools:  # Only for the first step
                # Skip reasoning entirely if mode is "none"
                if reasoning_mode != "none":
                    # Get the appropriate reasoning prompt based on verbosity mode
                    reasoning_prompt = reasoning_prompts.get(reasoning_mode, reasoning_prompts["medium"])

                    try:
                        # Get the reasoning
                        reasoning_start_time = time.time()
                        reasoning = self.llm.generate(reasoning_prompt)
                        reasoning_latency = time.time() - reasoning_start_time

                        # Only show reasoning if verbose mode is enabled
                        if verbose:
                            print(f"🧠 Reasoning: {reasoning}")

                        # Always store the full reasoning in memory and state
                        self.memory.add_to_history({
                            "role": "assistant",
                            "content": f"Reasoning: {reasoning}",
                            "type": "reasoning",
                            "timestamp": datetime.now().isoformat()
                        })

                        self.state.set("execution.initial_reasoning", reasoning)

                        # Log reasoning to audit trail if available
                        if hasattr(self, 'audit_trail') and self.audit_trail:
                            self.audit_trail.log_reasoning(
                                reasoning,
                                step=step,
                                metadata={
                                    "reasoning_mode": reasoning_mode,
                                    "latency": reasoning_latency
                                }
                            )
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error getting reasoning: {str(e)}")

                        # Log error to audit trail if available
                        if hasattr(self, 'audit_trail') and self.audit_trail:
                            self.audit_trail.log_error(
                                "reasoning_error",
                                str(e),
                                context={"step": step, "reasoning_mode": reasoning_mode}
                            )

            # Generate a response with potential tool usage
            if self.tools:
                # Generate a response that might include tool calls
                if isinstance(self.llm, (OpenAILLM, CustomOpenAILLM)):
                    response = self.llm.generate_with_tools(conversation, self.tools)
                else:
                    # For other LLMs, convert to text format
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
                    response = self.llm.generate_with_tools(prompt, self.tools)

                if response["type"] == "tool_call":
                    # The LLM wants to use a tool
                    tool_name = response["tool"]
                    tool_args = response["args"]
                    tool_reasoning = response.get("reasoning", "No explicit reasoning provided")

                    # Create a unique key for this tool call
                    tool_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

                    # Check for bypass_cache and force_repeat parameters
                    bypass_cache = False
                    force_repeat = False
                    if "bypass_cache" in tool_args:
                        bypass_cache = bool(tool_args.pop("bypass_cache"))
                    if "force_repeat" in tool_args:
                        force_repeat = bool(tool_args.pop("force_repeat"))

                    # Check if this exact tool call has already been made
                    duplicate_call = False
                    if not allow_repeated_calls and not force_repeat:
                        tool = next((t for t in self.tools if t.name == tool_name), None)
                        if tool and not getattr(tool, 'allow_repeated_calls', False):
                            for prev_call in self.tool_call_history:
                                if prev_call["tool"] == tool_name and self._compare_args(prev_call["args"], tool_args):
                                    duplicate_call = True
                                    if verbose:
                                        print(f"⚠️ Duplicate tool call detected: {tool_name}")

                                    # Instead of making the duplicate call, add a hint to use different tools
                                    conversation.append({
                                        "role": "system",
                                        "content": f"You've already called {tool_name} with these parameters. Please use the information you already have or try a different approach. If you genuinely need fresh data, include 'force_repeat': true in your parameters."
                                    })

                                    # Log duplicate call to audit trail if available
                                    if hasattr(self, 'audit_trail') and self.audit_trail:
                                        self.audit_trail.log_custom("duplicate_tool_call", {
                                            "tool": tool_name,
                                            "args": tool_args,
                                            "step": step
                                        })
                                    break

                    if duplicate_call:
                        # Track in state
                        self.state.append_to("execution.duplicate_calls", {
                            "tool": tool_name,
                            "args": tool_args,
                            "step": step,
                            "timestamp": datetime.now().isoformat()
                        })

                        step += 1
                        continue

                    # Find the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)

                    if tool:
                        # Use the cache if available and not bypassed
                        cache_to_use = None if bypass_cache else active_cache

                        # Execute the tool (or get cached result)
                        tool_start_time = time.time()
                        tool_result = tool(cache=cache_to_use, **tool_args)
                        tool_latency = time.time() - tool_start_time

                        # Record in tool call history to prevent repetition
                        self.tool_call_history.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                            "timestamp": datetime.now().isoformat()
                        })

                        # Track in state
                        self.state.set(f"tools.called.{tool_key}", {
                            "result": tool_result,
                            "timestamp": datetime.now().isoformat(),
                            "cached": tool_result.get("cached", False),
                            "step": step
                        })

                        # Log tool call to audit trail if available
                        if hasattr(self, 'audit_trail') and self.audit_trail:
                            self.audit_trail.log_tool_call(
                                tool_name,
                                tool_args,
                                tool_result,
                                latency=tool_latency,
                                cached=tool_result.get("cached", False),
                                metadata={"step": step}
                            )

                        # Extract information based on tool type
                        self._extract_info_from_tool_result(tool_name, tool_args, tool_result)

                        # Check if we got a cached result
                        if verbose:
                            print(f"🛠️ Tool selected: {tool_name}")
                            print(f"🔍 Parameters: {json.dumps(tool_args, indent=2)}")
                            print(f"🧠 Tool selection reasoning: {tool_reasoning}")
                            if tool_result.get("cached", False):
                                print("♻️ Using cached result")

                        # Add the reasoning to memory
                        self.memory.add_to_history({
                            "role": "assistant",
                            "content": f"Tool reasoning: {tool_reasoning}",
                            "type": "tool_reasoning",
                            "timestamp": datetime.now().isoformat()
                        })

                        # Add the tool call to the conversation and memory
                        cache_status = " (requesting fresh data)" if bypass_cache else ""
                        tool_call_message = f"I'll use the {tool_name} tool{cache_status} with these parameters: {json.dumps(tool_args)}"
                        conversation.append({
                            "role": "assistant",
                            "content": tool_call_message
                        })

                        self.memory.add_to_history({
                            "role": "assistant",
                            "content": tool_call_message,
                            "type": "tool_call",
                            "tool": tool_name,
                            "args": tool_args,
                            "timestamp": datetime.now().isoformat()
                        })

                        # Add cache status to tool result message
                        cache_notice = " (from cache)" if tool_result.get("cached", False) else ""

                        # Handle success or error in tool result
                        if isinstance(tool_result, dict) and "status" in tool_result:
                            if tool_result["status"] == "success":
                                result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result['result'])}"
                                if verbose:
                                    print(f"📊 Result: {tool_result['result']}")
                            else:  # Error case
                                error_msg = tool_result.get("error", "Unknown error")
                                result_content = f"Tool {tool_name} failed with error: {error_msg}"
                                if verbose:
                                    print(f"❌ Error: {error_msg}")

                                # Track error in state
                                self.state.set(f"tools.errors.{tool_key}", {
                                    "error": error_msg,
                                    "timestamp": datetime.now().isoformat(),
                                    "step": step
                                })

                                # Log error to audit trail if available
                                if hasattr(self, 'audit_trail') and self.audit_trail:
                                    self.audit_trail.log_error(
                                        "tool_execution_error",
                                        error_msg,
                                        context={"tool": tool_name, "args": tool_args, "step": step}
                                    )
                        else:
                            # Legacy format where the tool returns the result directly
                            result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result)}"
                            if verbose:
                                print(f"📊 Result: {tool_result}")

                        # Add the tool result to the conversation
                        conversation.append({
                            "role": "system",
                            "content": result_content
                        })

                        # Record in memory
                        self.memory.add_to_history({
                            "role": "system",
                            "content": result_content,
                            "type": "tool_result",
                            "cached": tool_result.get("cached", False),
                            "timestamp": datetime.now().isoformat()
                        })

                        # After getting the tool result, ask the model to reflect on what it learned
                        if step < max_steps - 1:  # Don't do this for the final step to save tokens
                            try:
                                # Adjust reflection level based on reasoning mode
                                if reasoning_mode == "short":
                                    reflection_prompt = [
                                        {"role": "system",
                                         "content": "In 1-2 sentences, what did you learn from this result?"},
                                        {"role": "user",
                                         "content": f"Tool: {tool_name}, Result summary: {str(tool_result)[:100]}..."}
                                    ]
                                elif reasoning_mode == "medium":
                                    reflection_prompt = [
                                        {"role": "system",
                                         "content": "In 3-6 sentences, how does this result help answer the query?"},
                                        {"role": "user",
                                         "content": f"Original query: {user_query}\nTool: {tool_name}\nResult summary: {str(tool_result)[:200]}..."}
                                    ]
                                else:  # "full"
                                    reflection_prompt = [
                                        {"role": "system",
                                         "content": "Based on the tool's result, reflect on what you've learned and how it helps answer the original query."},
                                        {"role": "user",
                                         "content": f"Original query: {user_query}\nTool used: {tool_name}\nTool result: {tool_result}\n\nReflect on what you've learned:"}
                                    ]

                                reflection_start_time = time.time()
                                reflection = self.llm.generate(reflection_prompt)
                                reflection_latency = time.time() - reflection_start_time

                                if verbose:
                                    print(f"🧠 Reflection: {reflection}")

                                # Store reflection in memory and state
                                self.memory.add_to_history({
                                    "role": "assistant",
                                    "content": f"Reflection: {reflection}",
                                    "type": "reflection",
                                    "timestamp": datetime.now().isoformat()
                                })

                                # Track reflection in state
                                self.state.append_to("execution.reflections", {
                                    "tool": tool_name,
                                    "content": reflection,
                                    "step": step,
                                    "timestamp": datetime.now().isoformat()
                                })

                                # Log reflection to audit trail if available
                                if hasattr(self, 'audit_trail') and self.audit_trail:
                                    self.audit_trail.log_custom("reflection", {
                                        "content": reflection,
                                        "tool": tool_name,
                                        "step": step,
                                        "latency": reflection_latency
                                    })
                            except Exception as e:
                                if verbose:
                                    print(f"⚠️ Error getting reflection: {str(e)}")

                                # Log error to audit trail if available
                                if hasattr(self, 'audit_trail') and self.audit_trail:
                                    self.audit_trail.log_error(
                                        "reflection_error",
                                        str(e),
                                        context={"tool": tool_name, "step": step}
                                    )

                        step += 1
                        continue

                elif response["type"] == "error":
                    # An error occurred during tool execution
                    error_message = response["content"]

                    # Add the error to the conversation
                    conversation.append({
                        "role": "system",
                        "content": error_message
                    })

                    # Record in memory and state
                    self.memory.add_to_history({
                        "role": "system",
                        "content": error_message,
                        "type": "error",
                        "timestamp": datetime.now().isoformat()
                    })

                    self.state.append_to("execution.errors", {
                        "message": error_message,
                        "step": step,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Log error to audit trail if available
                    if hasattr(self, 'audit_trail') and self.audit_trail:
                        self.audit_trail.log_error(
                            "response_error",
                            error_message,
                            context={"step": step}
                        )

                    if verbose:
                        print(f"❌ Error: {error_message}")

                    step += 1
                    continue
                else:
                    # Direct text response
                    final_response = response["content"]
            else:
                # No tools, just get a text response
                response_start_time = time.time()
                if isinstance(self.llm, (OpenAILLM, CustomOpenAILLM)):
                    final_response = self.llm.generate_with_tools(conversation, [])["content"]
                else:
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
                    final_response = self.llm.generate(prompt)
                response_latency = time.time() - response_start_time

            # Add the final response to conversation and memory
            conversation.append({
                "role": "assistant",
                "content": final_response
            })

            self.memory.add_to_history({
                "role": "assistant",
                "content": final_response,
                "type": "final_response",
                "timestamp": datetime.now().isoformat()
            })

            # Store final response in state
            self.state.set("response", {
                "content": final_response,
                "timestamp": datetime.now().isoformat(),
                "steps_taken": step + 1
            })

            # Log final response to audit trail if available
            if hasattr(self, 'audit_trail') and self.audit_trail:
                self.audit_trail.log_custom("final_response", {
                    "content": final_response,
                    "steps_taken": step + 1
                })

            if verbose:
                print(f"🤖 Final response: {final_response}")

            # We got a final answer, so we're done
            break

        # Increment step counter
        step += 1

        # Update step in state
        self.state.set("execution.total_steps", step)

        # Log run completion to audit trail if available
        if hasattr(self, 'audit_trail') and self.audit_trail:
            run_duration = time.time() - run_start_time
            self.audit_trail.log_custom("run_complete", {
                "steps_taken": step,
                "duration": run_duration,
                "token_usage": self.llm.get_token_usage()
            })

            # Save the audit trail to file if storage_path is set
            if hasattr(self.audit_trail, 'storage_path') and self.audit_trail.storage_path:
                audit_file = self.audit_trail.save()
                if verbose:
                    print(f"📝 Audit trail saved to: {audit_file}")

        # Prepare the return value based on requested format
        if return_trace:
            # Return structured response with trace
            return {
                "response": final_response,
                "history": self.memory.history,
                "trace": self.trace.format_trace(),
                "unique_tool_results": self.memory.get_unique_tool_results(),
                "state": self.state.to_dict(),
                "audit_trail": self.audit_trail.get_summary() if hasattr(self,
                                                                         'audit_trail') and self.audit_trail else None
            }
        else:
            # Return simple response and history
            return final_response, self.memory.history






# REPLACE the ConversationalAgent class in your agent.py with this enhanced version

class ConversationalAgent(Agent):
    """Agent specialized for multi-turn conversational interactions with audit trail support."""

    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100,
                 use_vector_memory: bool = False, enable_auditing: bool = True):
        
        # FIXED: Pass enable_auditing to parent class
        super().__init__(llm, system_prompt, cache_ttl, cache_size, enable_auditing)
        
        self.conversation = ConversationChain(
            llm,
            EnhancedMemory(),  # Already uses enhanced memory!
            system_prompt or "You are a helpful assistant that remembers previous interactions."
        )

        if use_vector_memory:
            self.init_vector_memory()

    def init_vector_memory(self):
        """Initialize vector-based semantic memory (placeholder for future vector DB integration)."""
        print("Vector memory initialized (using enhanced keyword search)")

    def chat(self, message: str, thread_id: str = "default",
             use_search: bool = True, context_window: int = 10) -> str:
        """Chat with the agent, maintaining conversation history with audit trail logging."""
        
        # Log the incoming chat to audit trail
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("chat_message_received", {
                "thread_id": thread_id,
                "message": message,
                "use_search": use_search,
                "context_window": context_window
            })
        
        # First try to use tools if needed
        if self.tools and len(self.tools) > 0:
            # Create a message format that includes history and the current message
            messages = self.conversation.get_conversation_context(thread_id, context_window)
            messages.append({"role": "user", "content": message})

            # Try to see if tools are needed
            response = self.llm.generate_with_tools(messages, self.tools)

            # Add user message to conversation history
            self.conversation.add_message("user", message, thread_id)

            # If the model wants to use a tool
            if response["type"] == "tool_call":
                tool_name = response["tool"]
                tool_args = response["args"]
                tool_reasoning = response.get("reasoning", "")

                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)

                if tool:
                    # Execute the tool
                    cache_to_use = self.cache if hasattr(self, 'cache') else None
                    tool_start_time = time.time()
                    tool_result = tool(cache=cache_to_use, **tool_args)
                    tool_latency = time.time() - tool_start_time

                    # Log tool call to audit trail
                    if hasattr(self, 'audit_trail') and self.audit_trail:
                        self.audit_trail.log_tool_call(
                            tool_name,
                            tool_args,
                            tool_result,
                            latency=tool_latency,
                            cached=tool_result.get("cached", False),
                            metadata={"thread_id": thread_id, "context": "conversational_chat"}
                        )

                    # Record the tool call
                    self.conversation.add_message(
                        "assistant",
                        f"I'll use the {tool_name} tool with these parameters: {json.dumps(tool_args)}",
                        thread_id
                    )

                    if tool_reasoning:
                        self.conversation.add_message(
                            "system",
                            f"Tool reasoning: {tool_reasoning}",
                            thread_id,
                            {"type": "reasoning"}
                        )
                        
                        # Log reasoning to audit trail
                        if hasattr(self, 'audit_trail') and self.audit_trail:
                            self.audit_trail.log_reasoning(
                                tool_reasoning,
                                metadata={"tool": tool_name, "thread_id": thread_id}
                            )

                    # Record the tool result
                    cache_notice = " (from cache)" if tool_result.get("cached", False) else ""
                    result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result.get('result', tool_result))}"

                    self.conversation.add_message("system", result_content, thread_id)

                    # Generate final answer using the tool result
                    messages = self.conversation.get_conversation_context(thread_id, context_window + 3)
                    messages.append({
                        "role": "system",
                        "content": "Based on the tool results above, provide a helpful response to the user."
                    })

                    response_start_time = time.time()
                    final_response = self.llm.generate(messages)
                    response_latency = time.time() - response_start_time
                    
                    self.conversation.add_message("assistant", final_response, thread_id)

                    # Log final response to audit trail
                    if hasattr(self, 'audit_trail') and self.audit_trail:
                        self.audit_trail.log_custom("conversational_response", {
                            "thread_id": thread_id,
                            "response": final_response,
                            "tool_used": tool_name,
                            "latency": response_latency
                        })

                    return final_response

            # If we didn't use a tool, continue with normal conversation flow

        # Normal conversation flow (no tools needed)
        response_start_time = time.time()
        if use_search:
            response = self.conversation.search_and_respond(
                message, thread_id, context_window
            )
        else:
            response = self.conversation.generate_response(
                message, thread_id, context_window
            )
        response_latency = time.time() - response_start_time
        
        # Log conversation response to audit trail
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("conversational_response", {
                "thread_id": thread_id,
                "user_message": message,
                "assistant_response": response,
                "use_search": use_search,
                "context_window": context_window,
                "latency": response_latency,
                "tool_used": None
            })
        
        return response

    def start_new_conversation(self, thread_id: str = None) -> str:
        """Start a new conversation thread with audit logging."""
        if thread_id is None:
            thread_id = f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create the new thread
        self.conversation.memory.threads[thread_id] = []
        
        # Log conversation start to audit trail
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("conversation_started", {
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            })

        return thread_id

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversation threads with summary info."""
        result = []

        for thread_id, indices in self.conversation.memory.threads.items():
            # Get first and last message timestamp
            if indices:
                first_msg = self.conversation.memory.history[indices[0]]
                last_msg = self.conversation.memory.history[indices[-1]]

                first_timestamp = first_msg.get("timestamp", "")
                last_timestamp = last_msg.get("timestamp", "")

                # Get a summary
                summary = self.conversation.summarize_conversation(thread_id)

                result.append({
                    "thread_id": thread_id,
                    "message_count": len(indices),
                    "first_message": first_timestamp,
                    "last_message": last_timestamp,
                    "summary": summary
                })

        return result

    def get_conversation_history(self, thread_id: str = "default",
                                 formatted: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """Get the conversation history for a thread."""
        history = self.conversation.memory.get_thread(thread_id)

        if formatted:
            formatted_history = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    formatted_history.append(f"User: {content}")
                elif role == "assistant":
                    formatted_history.append(f"Assistant: {content}")
                elif role == "system":
                    if "Tool result" in content:
                        formatted_history.append(f"[Tool Result] {content.replace('Tool result:', '')}")

            return "\n\n".join(formatted_history)

        return history

    def remember(self, key: str, value: Any, thread_id: str = "default") -> None:
        """Explicitly store information in memory with audit logging."""
        # Store in enhanced memory
        self.conversation.memory.set(key, value)
        
        # Add a system note about this storage
        self.conversation.add_message(
            "system",
            f"Remembered: {key} = {str(value)}",
            thread_id,
            {"type": "memory_storage", "key": key}
        )
        
        # Log to audit trail
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("memory_storage", {
                "key": key,
                "value": str(value),
                "thread_id": thread_id
            })

    def recall(self, key: str, default: Any = None) -> Any:
        """Recall information from memory."""
        value = self.conversation.memory.get(key, default)
        
        # Log recall to audit trail
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("memory_recall", {
                "key": key,
                "value": str(value) if value is not None else None,
                "found": value is not None
            })
        
        return value

    def search_memory(self, query: str, top_k: int = 5, thread_id: str = None) -> List[Dict[str, Any]]:
        """Search memory for relevant information with audit logging."""
        results = self.conversation.memory.find_relevant(query, top_k)
        
        # Log search to audit trail
        if hasattr(self, 'audit_trail') and self.audit_trail:
            self.audit_trail.log_custom("memory_search", {
                "query": query,
                "top_k": top_k,
                "results_count": len(results),
                "thread_id": thread_id
            })
        
        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        total_messages = len(self.conversation.memory.history)
        total_threads = len(self.conversation.memory.threads)
        
        # Count message types
        type_counts = {}
        for msg in self.conversation.memory.history:
            msg_type = msg.get("type", msg.get("role", "unknown"))
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        
        # Get indexed words count
        indexed_words = len(self.conversation.memory.indexed_content)
        
        return {
            "total_messages": total_messages,
            "total_threads": total_threads,
            "message_types": type_counts,
            "indexed_words": indexed_words,
            "has_audit_trail": hasattr(self, 'audit_trail') and self.audit_trail is not None
        }

    def start_audit_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new audit session for this conversational agent."""
        if hasattr(self, 'audit_trail') and self.audit_trail:
            session_metadata = {
                "agent_type": "ConversationalAgent",
                "system_prompt": self.system_prompt,
                "tools_available": [tool.name for tool in self.tools],
                **(metadata or {})
            }
            return self.audit_trail.start_session(session_metadata)
        return None

    def get_audit_summary(self) -> Optional[Dict[str, Any]]:
        """Get audit trail summary with conversation-specific metrics."""
        if hasattr(self, 'audit_trail') and self.audit_trail:
            summary = self.audit_trail.get_summary()
            
            # Add conversation-specific metrics
            memory_stats = self.get_memory_stats()
            summary["conversation_metrics"] = {
                "total_threads": memory_stats["total_threads"],
                "total_messages": memory_stats["total_messages"],
                "message_types": memory_stats["message_types"]
            }
            
            return summary
        return None

class EnhancedAgent(Agent):
    """Agent with explicit planning capabilities."""

    def __init__(self, llm, system_prompt=None, cache_ttl=3600, cache_size=100):
        super().__init__(llm, system_prompt, cache_ttl, cache_size)
        self.state = State()  # Initialize state
        self.planner = Planner(llm, self.tools, self.state)

    def plan_and_execute(self, goal: str, context: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """Generate and execute a plan to achieve a goal."""
        # Create a plan
        plan = self.planner.create_plan(goal, context)

        if verbose:
            actionable_steps = plan.get_actionable_steps()
            print(f"📝 Plan created with {len(actionable_steps)} actionable steps for goal: {goal}")
            print("-" * 50)
            for i, step in enumerate(actionable_steps):
                print(f"Step {i + 1}: {step['description']}")
                if step.get("tool"):
                    print(f"  Tool: {step['tool']}")
                    print(f"  Parameters: {json.dumps(step.get('params', {}))}")
                if step.get("depends_on"):
                    print(f"  Depends on: {step['depends_on']}")
                print()

            print(plan.visualize())

        # Execute the plan
        execution_result = self.planner.execute_plan(plan, verbose)

        # Store the executed plan in state
        self.state.set("plan.current", plan.to_dict())
        self.state.set("plan.execution_result", execution_result)

        # Generate a summary of the execution
        summary_prompt = f"""
        I executed a plan to achieve this goal: "{goal}"

        The steps and results were:
        {json.dumps(execution_result, indent=2)}

        Please provide a concise summary of the execution results, highlighting key findings
        and whether the goal was achieved. Focus on the actual data and insights discovered,
        not the process itself.
        """

        summary = self.llm.generate(summary_prompt)

        if verbose:
            print("\n📊 Execution Summary:")
            print("-" * 50)
            print(summary)

        return {
            "goal": goal,
            "plan": plan.to_dict(),
            "execution_result": execution_result,
            "summary": summary
        }

class AgentRole:
    """Defines a specialized role for an agent in a multi-agent system."""

    def __init__(self, name: str, description: str, system_prompt: str):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt


class AgentMessage:
    """Represents a message in a multi-agent conversation."""

    def __init__(self, sender: str, receiver: str, content: str,
                 message_type: str = "message", metadata: Dict[str, Any] = None):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type  # "message", "question", "proposal", "critique", etc.
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class MultiAgentSystem:
    """A system for coordinating multiple agents with different roles."""

    def __init__(self, llm: LLM, shared_state: State = None):
        self.llm = llm
        self.agents = {}  # {name: Agent}
        self.roles = {}  # {name: AgentRole}
        self.conversation = []  # List of AgentMessage objects
        self.state = shared_state or State()

    def add_agent(self, name: str, role: AgentRole, tools: List[Tool] = None) -> None:
        """Add an agent with a specific role to the system."""
        agent = Agent(self.llm, role.system_prompt)

        # Add tools if provided
        if tools:
            agent.add_tools(tools)

        self.agents[name] = agent
        self.roles[name] = role

    def broadcast_message(self, sender: str, content: str,
                          message_type: str = "message") -> None:
        """Send a message to all agents from the specified sender."""
        for receiver in self.agents.keys():
            if receiver != sender:
                self._send_message(sender, receiver, content, message_type)

    def _send_message(self, sender: str, receiver: str, content: str,
                      message_type: str = "message", metadata: Dict[str, Any] = None) -> None:
        """Send a message from one agent to another."""
        message = AgentMessage(sender, receiver, content, message_type, metadata)
        self.conversation.append(message)

    def _format_conversation_for_agent(self, agent_name: str) -> str:
        """Format the conversation history for a specific agent."""
        relevant_messages = []

        for msg in self.conversation:
            # Include messages where this agent is the sender or receiver
            # or broadcast messages where this agent is not the sender
            if msg.sender == agent_name or msg.receiver == agent_name or \
                    (msg.receiver == "all" and msg.sender != agent_name):
                if msg.sender == agent_name:
                    prefix = "You"
                else:
                    prefix = msg.sender

                relevant_messages.append(f"{prefix}: {msg.content}")

        return "\n\n".join(relevant_messages)

    def collaborative_solve(self, problem: str, max_turns: int = 10,
                            verbose: bool = False) -> Dict[str, Any]:
        """Have agents work together to solve a problem."""
        if not self.agents:
            raise ValueError("No agents in the system. Add agents first.")

        # Initialize the conversation with the problem
        self.conversation = []
        facilitator = "Facilitator"
        self._send_message(facilitator, "all",
                           f"Please solve this problem collaboratively: {problem}",
                           "problem")

        if verbose:
            print(f"🤝 Starting collaborative solving for problem: {problem}")
            print("-" * 50)

        # Track which agents have contributed in the current round
        for turn in range(max_turns):
            contributed = set()

            # Each agent takes a turn
            for agent_name, agent in self.agents.items():
                if agent_name in contributed:
                    continue

                # Prepare context for this agent
                context = self._format_conversation_for_agent(agent_name)
                agent_role = self.roles[agent_name]

                prompt = f"""
                You are {agent_name}, with this role: {agent_role.description}

                The group is trying to solve this problem: {problem}

                Conversation history:
                {context}

                Based on the conversation so far and your expertise, provide your contribution.
                You can:
                1. Share information or insights
                2. Ask questions to other agents
                3. Propose a solution or next step
                4. Critique or build upon others' ideas

                If you think the problem is solved, state that clearly.
                """

                # Get the agent's response
                response = agent.llm.generate(prompt)

                # Determine if this is a question, proposal, or regular message
                message_type = "message"
                if "?" in response and any(question_word in response.lower()
                                           for question_word in ["who", "what", "when", "where", "why", "how"]):
                    message_type = "question"
                elif any(proposal_word in response.lower()
                         for proposal_word in ["propose", "suggest", "recommend", "solution", "idea"]):
                    message_type = "proposal"

                # Send the message to all other agents
                self._send_message(agent_name, "all", response, message_type)
                contributed.add(agent_name)

                if verbose:
                    print(f"{agent_name}: {response}")
                    print("-" * 30)

                # Check if the agent thinks the problem is solved
                if "problem is solved" in response.lower() or "solution is complete" in response.lower():
                    # Have other agents confirm
                    confirmations = self._check_solution_consensus(problem, agent_name, response)

                    if all(confirmations.values()):
                        if verbose:
                            print("✅ All agents agree the problem is solved!")

                        # Generate final solution summary
                        return self._generate_solution_summary(problem)

            # If we've reached the maximum turns, generate a solution anyway
            if turn == max_turns - 1:
                if verbose:
                    print("⏰ Maximum turns reached. Generating solution summary.")

                return self._generate_solution_summary(problem)

        # Should not reach here if max_turns is positive
        return {"status": "error", "message": "No solution found"}

    def _check_solution_consensus(self, problem: str, proposer: str,
                                  proposed_solution: str) -> Dict[str, bool]:
        """Check if all agents agree with the proposed solution."""
        confirmations = {}

        for agent_name, agent in self.agents.items():
            if agent_name == proposer:
                confirmations[agent_name] = True
                continue

            prompt = f"""
            You are {agent_name}, with this role: {self.roles[agent_name].description}

            The original problem was: {problem}

            {proposer} has proposed this solution:
            {proposed_solution}

            Do you agree that this solution adequately solves the problem?
            Answer YES if you agree, or NO followed by your specific objections if you disagree.
            """

            response = agent.llm.generate(prompt)
            agrees = response.strip().upper().startswith("YES")

            confirmations[agent_name] = agrees

            # Add the confirmation or objection to the conversation
            message_type = "confirmation" if agrees else "objection"
            self._send_message(agent_name, "all", response, message_type)

        return confirmations

    def _generate_solution_summary(self, problem: str) -> Dict[str, Any]:
        """Generate a summary of the solution based on the conversation."""
        # Create a summary prompt for the LLM
        conversation_text = "\n\n".join([
            f"{msg.sender}: {msg.content}" for msg in self.conversation
        ])

        summary_prompt = f"""
        The following is a conversation between multiple agents trying to solve this problem:

        PROBLEM: {problem}

        CONVERSATION:
        {conversation_text}

        Please provide:
        1. A concise summary of the final solution
        2. The key contributions from each agent
        3. Any areas of disagreement or uncertainty

        Format your response as a structured report suitable for presentation.
        """

        # Use the LLM to generate a summary
        summary = self.llm.generate(summary_prompt)

        return {
            "problem": problem,
            "conversation": [msg.to_dict() for msg in self.conversation],
            "summary": summary
        }

    def debate(self, topic: str, rounds: int = 3, verbose: bool = False) -> Dict[str, Any]:
        """Have agents debate a topic with structured rounds."""
        if not self.agents:
            raise ValueError("No agents in the system. Add agents first.")
        if len(self.agents) < 2:
            raise ValueError("Debate requires at least two agents.")

        # Initialize the conversation with the topic
        self.conversation = []
        facilitator = "Facilitator"
        self._send_message(facilitator, "all",
                           f"Please debate this topic: {topic}",
                           "topic")

        if verbose:
            print(f"🎭 Starting debate on topic: {topic}")
            print("-" * 50)

        # Initial positions
        for agent_name, agent in self.agents.items():
            role = self.roles[agent_name]

            prompt = f"""
            You are {agent_name}, with this role: {role.description}

            The topic for debate is: {topic}

            Provide your initial position on this topic. Be clear and concise,
            focusing on your strongest 2-3 arguments.
            """

            position = agent.llm.generate(prompt)
            self._send_message(agent_name, "all", position, "position")

            if verbose:
                print(f"{agent_name} (Initial Position): {position}")
                print("-" * 30)

        # Debate rounds
        for round_num in range(rounds):
            if verbose:
                print(f"\nRound {round_num + 1}:")
                print("-" * 30)

            for agent_name, agent in self.agents.items():
                # Prepare context with all positions so far
                context = self._format_conversation_for_agent(agent_name)

                prompt = f"""
                You are {agent_name}, with this role: {self.roles[agent_name].description}

                The topic for debate is: {topic}

                This is round {round_num + 1} of the debate.

                Previous discussion:
                {context}

                Based on the discussion so far, provide your response. You should:
                1. Address the strongest counterarguments to your position
                2. Strengthen your main points with new evidence or reasoning
                3. Find points of agreement where possible
                """

                response = agent.llm.generate(prompt)
                self._send_message(agent_name, "all", response, "debate_round")

                if verbose:
                    print(f"{agent_name}: {response}")
                    print("-" * 30)

        # Have a neutral summarizer create a final summary
        conclusion_prompt = f"""
        The following is a debate between multiple perspectives on this topic:

        TOPIC: {topic}

        DEBATE TRANSCRIPT:
        {self._format_conversation_for_agent("summarizer")}

        As a neutral summarizer, please:
        1. Summarize the key points made by each side
        2. Identify the strongest arguments presented
        3. Note any areas of consensus or compromise
        4. Evaluate the quality of evidence and reasoning used

        Provide a balanced assessment that doesn't favor any particular position.
        """

        conclusion = self.llm.generate(conclusion_prompt)

        if verbose:
            print("\n📝 Debate Conclusion:")
            print(conclusion)

        return {
            "topic": topic,
            "rounds": rounds,
            "participants": list(self.agents.keys()),
            "conversation": [msg.to_dict() for msg in self.conversation],
            "conclusion": conclusion
        }
