import pytest
from unittest.mock import MagicMock, patch
from jerzy.llm import LLM, OpenAILLM
from jerzy.core import Tool

class TestLLM:
    def test_base_class(self):
        llm = LLM()
        assert llm.token_usage["prompt_tokens"] == 0
        assert llm.token_usage["completion_tokens"] == 0
        assert llm.token_usage["total_tokens"] == 0
        assert llm.token_usage["estimated_cost"] == 0.0
        assert len(llm.token_usage_history) == 0
        
        with pytest.raises(NotImplementedError):
            llm.generate("Test prompt")
        
        with pytest.raises(NotImplementedError):
            llm.generate_with_tools("Test prompt", [])
    
    def test_get_token_usage(self):
        llm = LLM()
        llm.token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "estimated_cost": 0.01
        }
        
        usage = llm.get_token_usage()
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert usage["estimated_cost"] == 0.01
    
    def test_reset_token_usage(self):
        llm = LLM()
        llm.token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "estimated_cost": 0.01
        }
        llm.token_usage_history = [{"some_entry": "value"}]
        
        llm.reset_token_usage()
        
        assert llm.token_usage["prompt_tokens"] == 0
        assert llm.token_usage["completion_tokens"] == 0
        assert llm.token_usage["total_tokens"] == 0
        assert llm.token_usage["estimated_cost"] == 0.0
        assert len(llm.token_usage_history) == 0

class TestOpenAILLM:
    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai_class):
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        assert llm.model == "gpt-4"
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)
    
    @patch('openai.OpenAI')
    def test_generate_string_prompt(self, mock_openai_class):
        # Setup mock client
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_usage = MagicMock()
        
        # Configure mocks
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        mock_message.content = "Response text"
        mock_completion.usage = mock_usage
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        # Test with string prompt
        response = llm.generate("Test prompt")
        
        # Check that client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test prompt"
        
        # Check response and token tracking
        assert response == "Response text"
        assert llm.token_usage["prompt_tokens"] == 10
        assert llm.token_usage["completion_tokens"] == 20
        assert llm.token_usage["total_tokens"] == 30
        assert len(llm.token_usage_history) == 1
    
    @patch('openai.OpenAI')
    def test_generate_message_list(self, mock_openai_class):
        # Setup mock client
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_usage = MagicMock()
        
        # Configure mocks
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        mock_message.content = "Response to messages"
        mock_completion.usage = mock_usage
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40
        
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        # Test with message list
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        response = llm.generate(messages)
        
        # Check that client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["messages"] == messages
        
        # Check response and token tracking
        assert response == "Response to messages"
        assert llm.token_usage["prompt_tokens"] == 15
        assert llm.token_usage["completion_tokens"] == 25
        assert llm.token_usage["total_tokens"] == 40
        assert len(llm.token_usage_history) == 1
    
    @patch('openai.OpenAI')
    def test_generate_with_tools_function_calling(self, mock_openai_class):
        # Setup mock client and tool
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        mock_function = MagicMock()
        mock_usage = MagicMock()
        
        # Configure mocks for tool calling
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        mock_message.tool_calls = [mock_tool_call]
        mock_tool_call.function = mock_function
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param1": "value1"}'
        mock_completion.usage = mock_usage
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 30
        mock_usage.total_tokens = 50
        
        # Create LLM and mock tool
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        def test_tool_func(param1: str) -> str:
            return f"Tool result: {param1}"
        
        # Create mock tool with ALL required attributes
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.signature = {"param1": {"type": "string", "required": True}}
        mock_tool.cacheable = True  # Add this attribute
        mock_tool.allow_repeated_calls = False  # Add this attribute
        mock_tool.return_value = "Tool result: value1"
        
        # Test tool calling
        result = llm.generate_with_tools("Use the test tool", [mock_tool])
        
        # Check that tool was called correctly
        assert result["type"] == "tool_call"
        assert result["tool"] == "test_tool"
        assert result["args"] == {"param1": "value1"}
        assert result["result"] == "Tool result: value1"
    
    @patch('openai.OpenAI')
    def test_generate_with_tools_text_response(self, mock_openai_class):
        # Setup mock client for text response
        mock_client = mock_openai_class.return_value
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_usage = MagicMock()
        
        # Configure mocks for text response
        mock_client.chat.completions.create.return_value = mock_completion
        mock_completion.choices = [mock_choice]
        mock_choice.message = mock_message
        mock_message.tool_calls = None
        mock_message.content = "Text response without tool call"
        mock_completion.usage = mock_usage
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15
        
        # Create LLM and mock tool
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        # Create mock tool with ALL required attributes 
        mock_tool = MagicMock()  # Don't use spec=Tool which restricts attributes
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"  # Add description
        mock_tool.signature = {"param1": {"type": "string", "required": True}}  # Add signature
        mock_tool.cacheable = True  # Add cacheable attribute
        mock_tool.allow_repeated_calls = False  # Add this attribute
        
        # Test text response
        result = llm.generate_with_tools("Don't use any tools", [mock_tool])
        
        # Should return text response
        assert result["type"] == "text"
        assert result["content"] == "Text response without tool call"
