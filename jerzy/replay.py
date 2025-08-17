# jerzy/replay.py - Deterministic replay system for testing LLM agents
"""
Cassette-based replay system for deterministic testing of LLM agents.
Records all LLM and tool calls, then replays them exactly for testing.
"""

import json
import hashlib
import os
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path


class Cassette:
    """Records and replays LLM/tool interactions for deterministic testing."""
    
    def __init__(self, cassette_path: str = None, mode: str = "auto"):
        """
        Initialize a cassette for recording/replaying interactions.
        
        Args:
            cassette_path: Path to cassette file (JSONL format)
            mode: "record" (always record), "replay" (only replay), "auto" (replay if exists, else record)
        """
        self.mode = mode
        self.cassette_path = cassette_path or "cassette.jsonl"
        self.interactions = self._load_cassette()
        self.session_metadata = {
            "start_time": datetime.now().isoformat(),
            "mode": mode,
            "hits": 0,
            "misses": 0
        }
    
    def _load_cassette(self) -> Dict[str, Any]:
        """Load existing cassette from file."""
        interactions = {}
        if os.path.exists(self.cassette_path):
            try:
                with open(self.cassette_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        interactions[entry["hash"]] = entry
            except Exception as e:
                print(f"Warning: Could not load cassette: {e}")
        return interactions
    
    def _save_interaction(self, interaction_hash: str, data: Dict[str, Any]) -> None:
        """Append a new interaction to the cassette file."""
        with open(self.cassette_path, 'a') as f:
            entry = {
                "hash": interaction_hash,
                "timestamp": datetime.now().isoformat(),
                **data
            }
            f.write(json.dumps(entry) + "\n")
    
    def _compute_hash(self, call_type: str, inputs: Dict[str, Any]) -> str:
        """Create deterministic hash for an interaction."""
        # Normalize the inputs for consistent hashing
        normalized = {
            "type": call_type,
            "inputs": json.dumps(inputs, sort_keys=True)
        }
        hash_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def intercept_llm_call(self, prompt: Union[str, list], model: str, **kwargs) -> str:
        """Intercept an LLM call - either replay from cassette or record new."""
        # Compute hash for this exact call
        inputs = {
            "prompt": prompt if isinstance(prompt, str) else json.dumps(prompt),
            "model": model,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'seed']}  # Ignore non-deterministic params
        }
        
        call_hash = self._compute_hash("llm", inputs)
        
        # Check if we should replay
        if self.mode in ["replay", "auto"] and call_hash in self.interactions:
            self.session_metadata["hits"] += 1
            recorded = self.interactions[call_hash]
            print(f"🎭 REPLAY: LLM call (hash: {call_hash[:8]}...)")
            return recorded["output"]
        
        # If replay mode but no match, error
        if self.mode == "replay":
            raise ValueError(f"Replay failed: No recorded interaction for hash {call_hash}")
        
        # We need to actually make the call (will be done by caller)
        self.session_metadata["misses"] += 1
        return None  # Signal to make real call
    
    def record_llm_response(self, prompt: Union[str, list], model: str, 
                           response: str, **kwargs) -> None:
        """Record an LLM response for future replay."""
        if self.mode in ["record", "auto"]:
            inputs = {
                "prompt": prompt if isinstance(prompt, str) else json.dumps(prompt),
                "model": model,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'seed']}
            }
            
            call_hash = self._compute_hash("llm", inputs)
            
            # Only save if not already recorded
            if call_hash not in self.interactions:
                data = {
                    "type": "llm",
                    "inputs": inputs,
                    "output": response
                }
                self._save_interaction(call_hash, data)
                self.interactions[call_hash] = {"hash": call_hash, **data}
                print(f"📼 RECORD: LLM call (hash: {call_hash[:8]}...)")
    
    def intercept_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Intercept a tool call - either replay from cassette or allow execution."""
        inputs = {
            "tool": tool_name,
            "args": args
        }
        
        call_hash = self._compute_hash("tool", inputs)
        
        # Check if we should replay
        if self.mode in ["replay", "auto"] and call_hash in self.interactions:
            self.session_metadata["hits"] += 1
            recorded = self.interactions[call_hash]
            print(f"🎭 REPLAY: Tool '{tool_name}' (hash: {call_hash[:8]}...)")
            return recorded["output"]
        
        # If replay mode but no match, error
        if self.mode == "replay":
            raise ValueError(f"Replay failed: No recorded tool call for {tool_name} with hash {call_hash}")
        
        self.session_metadata["misses"] += 1
        return None  # Signal to make real call
    
    def record_tool_response(self, tool_name: str, args: Dict[str, Any], 
                           result: Dict[str, Any]) -> None:
        """Record a tool response for future replay."""
        if self.mode in ["record", "auto"]:
            inputs = {
                "tool": tool_name,
                "args": args
            }
            
            call_hash = self._compute_hash("tool", inputs)
            
            # Only save if not already recorded
            if call_hash not in self.interactions:
                data = {
                    "type": "tool",
                    "inputs": inputs,
                    "output": result
                }
                self._save_interaction(call_hash, data)
                self.interactions[call_hash] = {"hash": call_hash, **data}
                print(f"📼 RECORD: Tool '{tool_name}' (hash: {call_hash[:8]}...)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        return {
            **self.session_metadata,
            "total_interactions": len(self.interactions),
            "cache_hit_rate": (
                self.session_metadata["hits"] / 
                max(1, self.session_metadata["hits"] + self.session_metadata["misses"])
            )
        }
    
    def clear(self) -> None:
        """Clear the cassette file and memory."""
        if os.path.exists(self.cassette_path):
            os.remove(self.cassette_path)
        self.interactions = {}
        print(f"🗑️ Cleared cassette: {self.cassette_path}")


class ReplayableAgent:
    """Mixin to add replay capabilities to any Agent class."""
    
    def enable_replay(self, cassette_path: str = None, mode: str = "auto") -> 'Cassette':
        """Enable cassette-based replay for this agent."""
        self.cassette = Cassette(cassette_path, mode)
        
        # Monkey-patch the LLM's generate method
        original_generate = self.llm.generate
        
        def replay_aware_generate(prompt, **kwargs):
            # Try to get from cassette first
            model = getattr(self.llm, 'model', 'unknown')
            replayed = self.cassette.intercept_llm_call(prompt, model, **kwargs)
            
            if replayed is not None:
                return replayed
            
            # Make real call
            response = original_generate(prompt, **kwargs)
            
            # Record for future replay
            self.cassette.record_llm_response(prompt, model, response, **kwargs)
            
            return response
        
        self.llm.generate = replay_aware_generate
        
        # Also patch generate_with_tools if it exists
        if hasattr(self.llm, 'generate_with_tools'):
            original_generate_with_tools = self.llm.generate_with_tools
            
            def replay_aware_generate_with_tools(prompt, tools, **kwargs):
                # For now, just call original - full tool replay is complex
                return original_generate_with_tools(prompt, tools, **kwargs)
            
            self.llm.generate_with_tools = replay_aware_generate_with_tools
        
        print(f"🎬 Replay enabled: mode='{mode}', cassette='{cassette_path or 'cassette.jsonl'}'")
        return self.cassette
