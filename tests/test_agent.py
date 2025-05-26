import pytest
from unittest.mock import MagicMock, patch
from jerzy.agent import Agent, ConversationalAgent, EnhancedAgent, MultiAgentSystem, AgentRole, AgentMessage
from jerzy.llm import LLM
from jerzy.memory import Memory
from jerzy.core import Tool, State
from jerzy.trace import AuditTrail

class TestAgent:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLM)
        llm.generate.return_value = "This is a response from the LLM"
        llm.generate_with_tools.return_value = {"type": "text", "content": "This is a response"}
        return llm
    
    @pytest.fixture
    def simple_agent(self, mock_llm):
        return Agent(mock_llm, "You are a helpful assistant.")
    
    def test_agent_initialization(self, mock_llm):
        agent = Agent(mock_llm, "Test prompt")
        assert agent.system_prompt == "Test prompt"
        assert agent.llm == mock_llm
        assert isinstance(agent.memory, Memory)
        assert isinstance(agent.tools, list)
    
    def test_add_tools(self, simple_agent):
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "test_tool"
        
        simple_agent.add_tools([mock_tool])
        assert len(simple_agent.tools) == 1
        assert simple_agent.tools[0].name == "test_tool"
        
        # Adding the same tool again shouldn't duplicate
        simple_agent.add_tools([mock_tool])
        assert len(simple_agent.tools) == 1
    
    def test_run_simple(self, simple_agent, mock_llm):
        response, history = simple_agent.run("Test query")
        assert response == "This is a response from the LLM"
        assert len(history) > 0
        mock_llm.generate.assert_called()
    
    def test_run_with_tools(self, simple_agent, mock_llm):
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.return_value = {"status": "success", "result": "Tool result"}
        
        simple_agent.add_tools([mock_tool])
        mock_llm.generate_with_tools.return_value = {
            "type": "tool_call", 
            "tool": "test_tool", 
            "args": {}, 
            "result": "tool result",
            "reasoning": "Test reasoning"
        }
        
        response, history = simple_agent.run("Test query with tool")
        assert isinstance(response, str)
        assert len(history) > 0

class TestConversationalAgent:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLM)
        llm.generate.return_value = "Conversation response"
        return llm
    
    @pytest.fixture
    def conv_agent(self, mock_llm):
        return ConversationalAgent(mock_llm, "Conversation prompt")
    
    def test_chat(self, conv_agent, mock_llm):
        response = conv_agent.chat("Hello")
        assert response == "Conversation response"
        mock_llm.generate.assert_called()
    
    def test_start_new_conversation(self, conv_agent):
        thread_id = conv_agent.start_new_conversation()
        assert thread_id is not None
        assert thread_id in conv_agent.conversation.memory.threads

class TestEnhancedAgent:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLM)
        # Set up the generate method to return valid JSON for planning
        llm.generate.return_value = """
        {
            "plan": [
                {
                    "description": "Test step",
                    "tool": "test_tool",
                    "params": {}
                }
            ]
        }
        """
        return llm
    
    @pytest.fixture
    def enhanced_agent(self, mock_llm):
        return EnhancedAgent(mock_llm)
    
    @patch('jerzy.trace.Planner')  # Update to patch from jerzy.trace instead of jerzy.agent
    def test_plan_and_execute(self, mock_planner_class, enhanced_agent, mock_llm):
        mock_planner = mock_planner_class.return_value
        mock_plan = MagicMock()
        mock_planner.create_plan.return_value = mock_plan
        mock_planner.execute_plan.return_value = {"status": "completed", "results": {}}
        
        # Set up the agent's planner attribute directly to ensure we're using our mock
        enhanced_agent.planner = mock_planner
        
        result = enhanced_agent.plan_and_execute("Test goal")
        
        assert "goal" in result
        assert "plan" in result
        assert "execution_result" in result
        assert "summary" in result
        mock_planner.create_plan.assert_called_once()
        mock_planner.execute_plan.assert_called_once()

class TestMultiAgentSystem:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock(spec=LLM)
        llm.generate.return_value = "MultiAgent response"
        return llm
    
    @pytest.fixture
    def multi_agent(self, mock_llm):
        return MultiAgentSystem(mock_llm)
    
    def test_add_agent(self, multi_agent):
        role = AgentRole("test_role", "Test description", "Test prompt")
        multi_agent.add_agent("agent1", role)
        
        assert "agent1" in multi_agent.agents
        assert "agent1" in multi_agent.roles
    
    def test_collaborative_solve(self, multi_agent, mock_llm):
        role1 = AgentRole("role1", "Description 1", "Prompt 1")
        role2 = AgentRole("role2", "Description 2", "Prompt 2")
        
        multi_agent.add_agent("agent1", role1)
        multi_agent.add_agent("agent2", role2)
        
        result = multi_agent.collaborative_solve("Test problem", max_turns=1)
        
        assert "problem" in result
        assert "conversation" in result
        assert "summary" in result
        assert mock_llm.generate.call_count >= 2  # At least called for each agent
