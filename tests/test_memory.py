import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
from jerzy.memory import Memory, EnhancedMemory

class TestMemory:
    def test_initialization(self):
        memory = Memory()
        assert isinstance(memory.storage, dict)
        assert isinstance(memory.history, list)
        assert isinstance(memory.tool_calls, list)
        assert isinstance(memory.reasoning_steps, list)
    
    def test_set_get(self):
        memory = Memory()
        
        # Set and get a simple value
        memory.set("key1", "value1")
        assert memory.get("key1") == "value1"
        
        # Get with default
        assert memory.get("nonexistent") is None
        assert memory.get("nonexistent", "default") == "default"
    
    def test_add_to_history(self):
        memory = Memory()
        
        # Add a regular entry
        entry1 = {"role": "user", "content": "Hello"}
        memory.add_to_history(entry1)
        
        # Add a tool call entry
        entry2 = {
            "role": "assistant", 
            "content": "Using search tool", 
            "type": "tool_call"
        }
        memory.add_to_history(entry2)
        
        # Add a reasoning entry
        entry3 = {
            "role": "assistant", 
            "content": "Reasoning: I should search for info", 
            "type": "reasoning"
        }
        memory.add_to_history(entry3)
        
        # Check that entries were added to the right places
        assert len(memory.history) == 3
        assert len(memory.tool_calls) == 1
        assert len(memory.reasoning_steps) == 1
        
        # Check that timestamps were added
        for entry in memory.history:
            assert "timestamp" in entry
    
    def test_get_history(self):
        memory = Memory()
        
        # Add various types of entries
        memory.add_to_history({"role": "user", "content": "User message", "type": "message"})
        memory.add_to_history({"role": "assistant", "content": "Tool use", "type": "tool_call"})
        memory.add_to_history({"role": "system", "content": "Result", "type": "result"})
        memory.add_to_history({"role": "assistant", "content": "Reasoning", "type": "reasoning"})
        memory.add_to_history({"role": "assistant", "content": "Response", "type": "message"})
        
        # Get all history
        all_history = memory.get_history()
        assert len(all_history) == 5
        
        # Get last n entries
        last_2 = memory.get_history(last_n=2)
        assert len(last_2) == 2
        assert last_2[0]["content"] == "Reasoning"
        assert last_2[1]["content"] == "Response"
        
        # Filter by entry type
        tool_calls = memory.get_history(entry_types=["tool_call"])
        assert len(tool_calls) == 1
        assert tool_calls[0]["content"] == "Tool use"
        
        # Combine filters
        filtered = memory.get_history(last_n=3, entry_types=["message"])
        assert len(filtered) == 2
        assert filtered[0]["content"] == "User message"
        assert filtered[1]["content"] == "Response"
    
    def test_get_unique_tool_results(self):
        memory = Memory()
        
        # Add tool results, including duplicates
        memory.add_to_history({
            "role": "system",
            "content": "Tool result: Result 1"
        })
        memory.add_to_history({
            "role": "system",
            "content": "Tool result: Result 2"
        })
        memory.add_to_history({
            "role": "system",
            "content": "Tool result: Result 1"  # Duplicate
        })
        memory.add_to_history({
            "role": "system",
            "content": "Tool result: Result 3"
        })
        
        unique_results = memory.get_unique_tool_results()
        assert len(unique_results) == 3
        assert "Result 1" in unique_results
        assert "Result 2" in unique_results
        assert "Result 3" in unique_results
        
        # Filter by tool name
        memory.add_to_history({
            "role": "system",
            "content": "Tool result: search_tool returned Result 4"
        })
        
        search_results = memory.get_unique_tool_results(tool_name="search_tool")
        assert len(search_results) == 1
        assert any("Result 4" in r for r in search_results)
    
    def test_get_last_reasoning(self):
        memory = Memory()
        
        # Add various entries
        memory.add_to_history({"role": "user", "content": "User message"})
        memory.add_to_history({"role": "assistant", "content": "Reasoning: First thought", "type": "reasoning"})
        memory.add_to_history({"role": "assistant", "content": "Response"})
        memory.add_to_history({"role": "assistant", "content": "Reasoning: Second thought", "type": "reasoning"})
        
        last_reasoning = memory.get_last_reasoning()
        assert last_reasoning == "Second thought"
        
        # Test when no reasoning exists
        empty_memory = Memory()
        assert empty_memory.get_last_reasoning() is None
    
    def test_get_reasoning_chain(self):
        memory = Memory()
        
        # Add various entries including multiple reasoning steps
        memory.add_to_history({"role": "user", "content": "User message"})
        memory.add_to_history({"role": "assistant", "content": "Reasoning: First step", "type": "reasoning"})
        memory.add_to_history({"role": "assistant", "content": "Response 1"})
        memory.add_to_history({"role": "assistant", "content": "Reasoning: Second step", "type": "reasoning"})
        memory.add_to_history({"role": "assistant", "content": "Response 2"})
        
        reasoning_chain = memory.get_reasoning_chain()
        assert len(reasoning_chain) == 2
        assert reasoning_chain[0] == "First step"
        assert reasoning_chain[1] == "Second step"

class TestEnhancedMemory:
    def test_initialization(self):
        memory = EnhancedMemory(max_history_length=50)
        
        assert memory.max_history_length == 50
        assert isinstance(memory.threads, dict)
        assert memory.current_thread == "default"
        assert isinstance(memory.indexed_content, dict)
    
    def test_add_to_thread(self):
        memory = EnhancedMemory()
        
        # Add messages to two different threads
        memory.add_to_thread("thread1", {
            "role": "user", 
            "content": "Message in thread 1"
        })
        
        memory.add_to_thread("thread2", {
            "role": "assistant", 
            "content": "Message in thread 2"
        })
        
        memory.add_to_thread("thread1", {
            "role": "assistant", 
            "content": "Reply in thread 1"
        })
        
        # Check thread contents
        assert "thread1" in memory.threads
        assert "thread2" in memory.threads
        assert len(memory.threads["thread1"]) == 2
        assert len(memory.threads["thread2"]) == 1
        
        # Check global history
        assert len(memory.history) == 3
        
        # Check indexing
        assert "message" in memory.indexed_content
        assert len(memory.indexed_content["message"]) == 2  # Present in both threads
        assert "reply" in memory.indexed_content
        assert len(memory.indexed_content["reply"]) == 1
    
    def test_get_thread(self):
        memory = EnhancedMemory()
        
        # Add messages to thread
        memory.add_to_thread("test_thread", {
            "role": "user", 
            "content": "Message 1"
        })
        
        memory.add_to_thread("test_thread", {
            "role": "assistant", 
            "content": "Reply 1"
        })
        
        memory.add_to_thread("test_thread", {
            "role": "user", 
            "content": "Message 2"
        })
        
        # Get full thread
        thread = memory.get_thread("test_thread")
        assert len(thread) == 3
        assert thread[0]["content"] == "Message 1"
        assert thread[1]["content"] == "Reply 1"
        assert thread[2]["content"] == "Message 2"
        
        # Get limited thread
        limited = memory.get_thread("test_thread", last_n=2)
        assert len(limited) == 2
        assert limited[0]["content"] == "Reply 1"
        assert limited[1]["content"] == "Message 2"
        
        # Get nonexistent thread
        assert memory.get_thread("nonexistent") == []
    
    def test_summarize_thread(self):
        memory = EnhancedMemory()
        
        # Add messages to thread
        memory.add_to_thread("test_thread", {
            "role": "user", 
            "content": "Tell me about AI"
        })
        
        memory.add_to_thread("test_thread", {
            "role": "assistant", 
            "content": "AI is a field of computer science..."
        })
        
        # Test without LLM (fallback)
        summary = memory.summarize_thread("test_thread")
        assert "Thread with 2 messages" in summary
        
        # Test with LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Summary of AI conversation"
        
        summary = memory.summarize_thread("test_thread", mock_llm)
        assert summary == "Summary of AI conversation"
        mock_llm.generate.assert_called_once()
        
        # Test empty thread
        summary = memory.summarize_thread("nonexistent")
        assert "No messages" in summary
    
    def test_prune_history(self):
        memory = EnhancedMemory(max_history_length=5)
        
        # Add 10 messages across 2 threads
        for i in range(5):
            memory.add_to_thread("thread1", {
                "role": "user", 
                "content": f"Thread 1 message {i}"
            })
            
            memory.add_to_thread("thread2", {
                "role": "user", 
                "content": f"Thread 2 message {i}"
            })
        
        assert len(memory.history) == 10
        
        # Now prune to keep only the last 6 messages
        memory.prune_history(keep_last_n=6)
        
        # Should have pruned 4 oldest messages
        assert len(memory.history) == 6
        assert memory.history[0]["content"] == "Thread 1 message 2"
        
        # Thread indices should be updated
        assert len(memory.threads["thread1"]) == 3  # Last 3 messages from thread 1
        assert len(memory.threads["thread2"]) == 3  # Last 3 messages from thread 2
        
        # First thread index should now be 0 instead of 4
        assert memory.threads["thread1"][0] == 0
    
    def test_find_relevant(self):
        memory = EnhancedMemory()
        
        # Add messages with varied content
        memory.add_to_thread("thread", {
            "role": "user", 
            "content": "Tell me about machine learning algorithms"
        })
        
        memory.add_to_thread("thread", {
            "role": "assistant", 
            "content": "Machine learning includes supervised and unsupervised learning"
        })
        
        memory.add_to_thread("thread", {
            "role": "user", 
            "content": "What about deep learning?"
        })
        
        memory.add_to_thread("thread", {
            "role": "assistant", 
            "content": "Deep learning is a subset of machine learning using neural networks"
        })
        
        memory.add_to_thread("thread", {
            "role": "user", 
            "content": "Explain neural networks"
        })
        
        # Search for relevant messages
        relevant = memory.find_relevant("How do neural networks relate to deep learning?", top_k=2)
        
        # Should match the messages containing neural networks and deep learning
        assert len(relevant) == 2
        assert any("neural networks" in msg["content"] for msg in relevant)
        assert any("deep learning" in msg["content"] for msg in relevant)
        
        # Search with no matches
        no_matches = memory.find_relevant("quantum computing", top_k=2)
        assert len(no_matches) == 0
    
    def test_save_and_load(self):
        # Create a memory with some content
        original_memory = EnhancedMemory()
        original_memory.add_to_thread("thread1", {
            "role": "user", 
            "content": "Original message"
        })
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name
        
        try:
            # Save memory to file
            original_memory.save_to_file(file_path)
            
            # Create a new memory and load from file
            loaded_memory = EnhancedMemory()
            loaded_memory.load_from_file(file_path)
            
            # Verify data was loaded correctly
            assert len(loaded_memory.history) == 1
            assert loaded_memory.history[0]["content"] == "Original message"
            assert "thread1" in loaded_memory.threads
            assert len(loaded_memory.threads["thread1"]) == 1
            
            # Verify indexing was rebuilt
            assert "original" in loaded_memory.indexed_content
            assert "message" in loaded_memory.indexed_content
            
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_load_from_invalid_file(self):
        memory = EnhancedMemory()
        
        # Try to load from nonexistent file
        with patch('jerzy.memory.print') as mock_print:
            memory.load_from_file("nonexistent_file.json")
            mock_print.assert_called()
