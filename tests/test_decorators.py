import pytest
from unittest.mock import patch, MagicMock
import logging
from jerzy.decorators import robust_tool, with_fallback, log_tool_call

class TestRobustTool:
    def test_retry_success_after_failure(self):
        # Mock function that fails twice then succeeds
        mock_func = MagicMock(side_effect=[ValueError("First error"), 
                                          ValueError("Second error"), 
                                          "success"])
        
        # Apply the decorator with 3 retries
        decorated = robust_tool(retries=3, wait_seconds=0.01)(mock_func)
        
        # Call the decorated function
        result = decorated()
        
        # Should eventually succeed
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_exhaustion(self):
        # Mock function that always fails
        mock_func = MagicMock(side_effect=ValueError("Always fails"))
        
        # Apply the decorator with 2 retries
        decorated = robust_tool(retries=2, wait_seconds=0.01)(mock_func)
        
        # Call should eventually fail after retries are exhausted
        with pytest.raises(Exception):
            decorated()
        
        # The tenacity.retry decorator with stop_after_attempt(2) will make
        # a total of 2 attempts (initial + 1 retry), not 3
        assert mock_func.call_count == 2
    
    def test_preserves_function_metadata(self):
        def test_func(param1):
            """Test docstring."""
            return param1
        
        decorated = robust_tool()(test_func)
        
        assert decorated.__name__ == "test_func"
        assert decorated.__doc__ == "Test docstring."

class TestWithFallback:
    def test_primary_success(self):
        def primary_func(param):
            return f"Primary: {param}"
        
        def fallback_func(param):
            return f"Fallback: {param}"
        
        decorated = with_fallback(fallback_func)(primary_func)
        
        # Should use primary function
        assert decorated("test") == "Primary: test"
    
    def test_primary_fails_fallback_used(self):
        def primary_func(param):
            raise ValueError("Primary failed")
        
        def fallback_func(param):
            return f"Fallback: {param}"
        
        decorated = with_fallback(fallback_func)(primary_func)
        
        # Primary fails, should use fallback
        assert decorated("test") == "Fallback: test"
    
    def test_preserves_function_metadata(self):
        def primary_func(param):
            """Primary docstring."""
            return param
        
        def fallback_func(param):
            return param
        
        decorated = with_fallback(fallback_func)(primary_func)
        
        assert decorated.__name__ == "primary_func"
        assert decorated.__doc__ == "Primary docstring."

class TestLogToolCall:
    @patch('jerzy.decorators.logging')
    @patch('jerzy.decorators.open')
    def test_logging_successful_call(self, mock_open, mock_logging):
        # Setup mock file handle
        file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = file_handle
        
        # Create a decorated function
        @log_tool_call("test_tool")
        def test_func(param):
            return {"result": param}
        
        # Call the function
        result = test_func("test_value")
        
        # Check result
        assert result == {"result": "test_value"}
        
        # Check that logging occurred
        mock_logging.info.assert_called_once()
        file_handle.write.assert_called_once()
        
        # Check log content contains expected fields
        log_content = file_handle.write.call_args[0][0]
        assert "test_tool" in log_content
        assert "success" in log_content
        assert "test_value" in log_content
    
    @patch('jerzy.decorators.logging')
    @patch('jerzy.decorators.open')
    def test_logging_failed_call(self, mock_open, mock_logging):
        # Setup mock file handle
        file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = file_handle
        
        # Create a decorated function that raises an error
        @log_tool_call("failing_tool")
        def failing_func(param):
            raise ValueError("Test error")
        
        # Call the function
        result = failing_func("test_value")
        
        # Check result contains error
        assert "error" in result
        assert "Test error" in result["error"]
        
        # Check that logging occurred
        mock_logging.info.assert_called_once()
        file_handle.write.assert_called_once()
        
        # Check log content contains expected fields
        log_content = file_handle.write.call_args[0][0]
        assert "failing_tool" in log_content
        assert "error" in log_content
