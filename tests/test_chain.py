import pytest
from unittest.mock import MagicMock
from jerzy.chain import Chain, ConversationChain
from jerzy.llm import LLM
from jerzy.memory import Memory, EnhancedMemory

class TestChain:
    def test_chain_initialization(self):
        chain = Chain()
        assert isinstance(chain.steps, list)
        assert len(chain.steps) == 0
        assert isinstance(chain.memory, Memory)
    
    def test_add_step(self):
        chain = Chain()
        
        def test_step(context, memory):
            context["test_key"] = "test_value"
            return context
        
        result = chain.add(test_step)
        
        assert len(chain.steps) == 1
        assert result == chain  # Should return self for chaining
    
    def test_execute(self):
        chain = Chain()
        
        def step1(context, memory):
            context["step1"] = "completed"
            memory.set("step1_completed", True)
            return context
        
        def step2(context, memory):
            context["step2"] = "completed"
            assert memory.get("step1_completed") == True
            return context
        
        chain.add(step1).add(step2)
        
        result = chain.execute({"initial": "value"})
        
        assert result["initial"] == "value"
        assert result["step1"] == "completed"
        assert result["step2"] == "completed"
        assert chain.memory.get("step1_completed") == True

class TestConversationChain:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLM)
        llm.generate.return_value = "Response from LLM"
        return llm
    
    @pytest.fixture
    def conversation_chain(self, mock_llm):
        return ConversationChain(mock_llm)
    
    def test_initialization(self, mock_llm):
        chain = ConversationChain(mock_llm, system_prompt="Test prompt")
        assert chain.llm == mock_llm
        assert chain.system_prompt == "Test prompt"
        assert isinstance(chain.memory, EnhancedMemory)
        assert isinstance(chain.state, dict) or hasattr(chain.state, "data")
    
    def test_add_message(self, conversation_chain):
        conversation_chain.add_message("user", "Hello", "thread1")
        
        # Verify message was added
        messages = conversation_chain.memory.get_thread("thread1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
    
    def test_get_conversation_context(self, conversation_chain):
        # Add some messages
        conversation_chain.add_message("user", "Message 1", "thread1")
        conversation_chain.add_message("assistant", "Response 1", "thread1")
        conversation_chain.add_message("user", "Message 2", "thread1")
        
        # Get context
        context = conversation_chain.get_conversation_context("thread1", 2)
        
        # Should include system prompt + last 2 messages
        assert len(context) == 3
        assert context[0]["role"] == "system"  # System prompt
        assert context[1]["content"] == "Response 1"
        assert context[2]["content"] == "Message 2"
    
    def test_generate_response(self, conversation_chain, mock_llm):
        response = conversation_chain.generate_response("Hello", "thread1")
        
        assert response == "Response from LLM"
        assert mock_llm.generate.called
        
        # Check that messages were added to memory
        messages = conversation_chain.memory.get_thread("thread1")
        assert len(messages) == 2  # User message and assistant response
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Response from LLM"
    
    def test_search_and_respond(self, conversation_chain, mock_llm):
        # Add some history first
        conversation_chain.add_message("user", "Previous question about Python", "thread1")
        conversation_chain.add_message("assistant", "Python is a programming language", "thread1")
        
        # Mock the find_relevant method
        original_find_relevant = conversation_chain.memory.find_relevant
        conversation_chain.memory.find_relevant = MagicMock(return_value=[
            {"role": "user", "content": "Previous question about Python"},
            {"role": "assistant", "content": "Python is a programming language"}
        ])
        
        try:
            response = conversation_chain.search_and_respond("Tell me more about Python", "thread1")
            
            assert response == "Response from LLM"
            assert conversation_chain.memory.find_relevant.called
            assert mock_llm.generate.called
            
            # Check that call to LLM included the relevant context
            args = mock_llm.generate.call_args[0][0]
            assert any("relevant context" in msg.get("content", "") for msg in args)
            
        finally:
            # Restore original method
            conversation_chain.memory.find_relevant = original_find_relevant
