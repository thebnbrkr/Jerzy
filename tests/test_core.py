import pytest
import time
from unittest.mock import MagicMock, patch
from jerzy.core import Prompt, ToolCache, State, Tool

class TestPrompt:
    def test_initialization(self):
        template = "Hello, {name}!"
        prompt = Prompt(template)
        assert prompt.template == template
    
    def test_format(self):
        prompt = Prompt("Hello, {name}! You are {age} years old.")
        formatted = prompt.format(name="Alice", age=30)
        assert formatted == "Hello, Alice! You are 30 years old."

class TestToolCache:
    @pytest.fixture
    def cache(self):
        return ToolCache(max_size=2, ttl=1)  # 1 second TTL for testing
    
    def test_initialization(self):
        cache = ToolCache(max_size=10, ttl=60)
        assert cache.max_size == 10
        assert cache.ttl == 60
        assert isinstance(cache.cache, dict)
        assert len(cache.cache) == 0
    
    def test_generate_key(self, cache):
        key1 = cache._generate_key("tool1", {"a": 1, "b": 2})
        key2 = cache._generate_key("tool1", {"b": 2, "a": 1})  # Same args, different order
        key3 = cache._generate_key("tool2", {"a": 1, "b": 2})  # Different tool
        
        assert key1 == key2  # Order shouldn't matter
        assert key1 != key3  # Different tool name should produce different key
    
    def test_set_and_get(self, cache):
        # Set a value
        tool_name = "test_tool"
        args = {"param1": "value1"}
        result = {"status": "success", "result": "test result"}
        
        cache.set(tool_name, args, result)
        
        # Get the value back
        cached_result = cache.get(tool_name, args)
        assert cached_result == result
    
    def test_ttl_expiration(self, cache):
        # Set a value with 1 second TTL
        tool_name = "test_tool"
        args = {"param1": "value1"}
        result = {"status": "success", "result": "test result"}
        
        cache.set(tool_name, args, result)
        
        # Verify it's in the cache
        assert cache.get(tool_name, args) == result
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Value should now be None
        assert cache.get(tool_name, args) is None
    
    def test_max_size(self):
        # Create cache with max size 2
        cache = ToolCache(max_size=2, ttl=None)
        
        # Add 3 items
        cache.set("tool1", {"a": 1}, {"result": "result1"})
        cache.set("tool2", {"b": 2}, {"result": "result2"})
        cache.set("tool3", {"c": 3}, {"result": "result3"})
        
        # Should only keep the most recent 2
        assert len(cache.cache) == 2
        assert cache.get("tool1", {"a": 1}) is None  # This should be evicted
        assert cache.get("tool2", {"b": 2}) is not None
        assert cache.get("tool3", {"c": 3}) is not None
    
    def test_clear(self, cache):
        cache.set("tool1", {"a": 1}, {"result": "result1"})
        cache.set("tool2", {"b": 2}, {"result": "result2"})
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
    
    def test_remove(self, cache):
        cache.set("tool1", {"a": 1}, {"result": "result1"})
        cache.set("tool2", {"b": 2}, {"result": "result2"})
        
        assert len(cache.cache) == 2
        
        # Remove one item
        cache.remove("tool1", {"a": 1})
        
        assert len(cache.cache) == 1
        assert cache.get("tool1", {"a": 1}) is None
        assert cache.get("tool2", {"b": 2}) is not None

class TestState:
    def test_initialization(self):
        state = State()
        assert isinstance(state.data, dict)
        assert isinstance(state.history, list)
        assert state.version == 0
    
    def test_set_simple(self):
        state = State()
        state.set("key1", "value1")
        
        assert state.data["key1"] == "value1"
        assert len(state.history) == 1
        assert state.version == 1
    
    def test_set_nested(self):
        state = State()
        state.set("parent.child.grandchild", "nested_value")
        
        # Check nested structure was created
        assert "parent" in state.data
        assert "child" in state.data["parent"]
        assert "grandchild" in state.data["parent"]["child"]
        assert state.data["parent"]["child"]["grandchild"] == "nested_value"
    
    def test_get_simple(self):
        state = State()
        state.set("key1", "value1")
        
        assert state.get("key1") == "value1"
        assert state.get("nonexistent") is None
        assert state.get("nonexistent", "default") == "default"
    
    def test_get_nested(self):
        state = State()
        state.set("parent.child.grandchild", "nested_value")
        
        assert state.get("parent.child.grandchild") == "nested_value"
        assert state.get("parent.child.nonexistent") is None
        assert state.get("parent.nonexistent.grandchild") is None
    
    def test_has_key(self):
        state = State()
        state.set("key1", "value1")
        state.set("parent.child", "child_value")
        
        assert state.has_key("key1") is True
        assert state.has_key("nonexistent") is False
        assert state.has_key("parent") is True
        assert state.has_key("parent.child") is True
        assert state.has_key("parent.nonexistent") is False
    
    def test_append_to(self):
        state = State()
        
        # First append creates list
        state.append_to("list_key", "item1")
        assert state.get("list_key") == ["item1"]
        
        # Subsequent appends add to list
        state.append_to("list_key", "item2")
        assert state.get("list_key") == ["item1", "item2"]
        
        # Appending to non-list converts to list
        state.set("string_key", "value")
        state.append_to("string_key", "appended")
        assert state.get("string_key") == ["value", "appended"]
    
    def test_to_dict(self):
        state = State()
        state.set("key1", "value1")
        state.set("parent.child", "child_value")
        
        state_dict = state.to_dict()
        
        assert state_dict["key1"] == "value1"
        assert state_dict["parent"]["child"] == "child_value"
        
        # Verify it's a copy, not a reference
        state_dict["key1"] = "modified"
        assert state.get("key1") == "value1"

class TestTool:
    def test_initialization(self):
        def test_func(param1: str, param2: int = 0) -> dict:
            return {"result": param1 * param2}
        
        tool = Tool("test_tool", test_func, "Test description", cacheable=True)
        
        assert tool.name == "test_tool"
        assert tool.func == test_func
        assert tool.description == "Test description"
        assert tool.cacheable is True
        assert isinstance(tool.signature, dict)
    
    def test_signature_extraction(self):
        def test_func(param1: str, param2: int = 0, *, param3: bool = True) -> dict:
            return {}
        
        tool = Tool("test_tool", test_func, "Description")
        
        # Check signature
        assert "param1" in tool.signature
        assert "param2" in tool.signature
        assert "param3" in tool.signature
        
        # Check types
        assert tool.signature["param1"]["type"] == "str"
        assert tool.signature["param2"]["type"] == "int"
        assert tool.signature["param3"]["type"] == "bool"
        
        # Check required flag
        assert tool.signature["param1"]["required"] is True
        assert tool.signature["param2"]["required"] is False
        assert tool.signature["param3"]["required"] is False
    
    def test_call_successful(self):
        def test_func(param1: str) -> str:
            return f"Result: {param1}"
        
        tool = Tool("test_tool", test_func, "Description")
        result = tool("test_value")
        
        assert result["status"] == "success"
        assert result["result"] == "Result: test_value"
        assert "timestamp" in result
        assert result["cached"] is False
    
    def test_call_with_error(self):
        def failing_func(param1: str) -> str:
            raise ValueError("Test error")
        
        tool = Tool("failing_tool", failing_func, "Description")
        result = tool("test_value")
        
        assert result["status"] == "error"
        assert "error" in result
        assert "Test error" in result["error"]
        assert result["error_type"] == "ValueError"
    
    def test_call_with_cache(self):
        mock_func = MagicMock(return_value="cached_result")
        tool = Tool("cache_tool", mock_func, "Description", cacheable=True)
        
        # Create a mock cache
        mock_cache = MagicMock(spec=ToolCache)
        mock_cache.get.return_value = {
            "status": "success", 
            "result": "cached_result", 
            "timestamp": "2023-01-01T00:00:00"
        }
        
        # Call with cache
        result = tool(cache=mock_cache, param1="test")
        
        # Should return cached result with cached flag set to True
        assert result["cached"] is True
        assert result["result"] == "cached_result"
        
        # Function should not be called, result from cache
        mock_func.assert_not_called()
        mock_cache.get.assert_called_once()
