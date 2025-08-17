# jerzy/replay.py - Fixed replay system with proper determinism
"""
Cassette-based replay system for deterministic testing of LLM agents.
Records all LLM and tool calls, then replays them exactly for testing.
"""

import json
import hashlib
import os
import functools
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from pathlib import Path


class Cassette:
    """Records and replays LLM/tool interactions for deterministic testing."""
    
    def __init__(self, cassette_path: str = None, mode: str = "auto", verbose: bool = False):
        """
        Initialize a cassette for recording/replaying interactions.
        
        Args:
            cassette_path: Path to cassette file (JSONL format)
            mode: "record" (always record), "replay" (only replay), "auto" (replay if exists, else record)
            verbose: Whether to print debug messages
        """
        self.mode = mode
        self.verbose = verbose
        self.cassette_path = cassette_path or os.environ.get('JERZY_CASSETTE', 'cassette.jsonl')
        self.interactions = self._load_cassette()
        self.session_metadata = {
            "start_time": datetime.now().isoformat(),
            "mode": mode,
            "llm_hits": 0,
            "llm_misses": 0,
            "tool_hits": 0,
            "tool_misses": 0
        }
    
    def _load_cassette(self) -> Dict[str, Any]:
        """Load existing cassette from file."""
        interactions = {}
        if os.path.exists(self.cassette_path):
            try:
                with open(self.cassette_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        # Skip metadata lines
                        if "_meta" in entry:
                            continue
                        interactions[entry["hash"]] = entry
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load cassette: {e}")
        return interactions
    
    def _save_interaction(self, interaction_hash: str, data: Dict[str, Any]) -> None:
        """Append a new interaction to the cassette file."""
        # Create file with metadata if it doesn't exist
        if not os.path.exists(self.cassette_path):
            with open(self.cassette_path, 'w') as f:
                meta = {"_meta": {"version": "1", "created": datetime.now().isoformat()}}
                f.write(json.dumps(meta) + "\n")
        
        with open(self.cassette_path, 'a') as f:
            entry = {
                "hash": interaction_hash,
                "timestamp": datetime.now().isoformat(),
                **data
            }
            f.write(json.dumps(entry) + "\n")
    
    def _stable_gen_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only stable generation parameters that affect output."""
        keep = ("temperature", "top_p", "frequency_penalty", "presence_penalty", 
                "seed", "stop", "max_tokens", "n")
        return {k: params[k] for k in keep if k in params}
    
    def _canonical_messages(self, msgs: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, str]]]:
        """Canonicalize messages to stable format for hashing."""
        if isinstance(msgs, str):
            return msgs
        
        canon = []
        for m in msgs:
            canon.append({
                "role": m.get("role"),
                "content": m.get("content")
            })
        return canon
    
    def _compute_hash(self, call_type: str, inputs: Dict[str, Any]) -> str:
        """Create deterministic hash for an interaction."""
        # Normalize the inputs for consistent hashing
        normalized = {
            "type": call_type,
            "inputs": json.dumps(inputs, sort_keys=True, ensure_ascii=True)
        }
        hash_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def intercept_generic(self, call_type: str, key_inputs: Dict[str, Any]) -> Optional[Any]:
        """Generic interception for any call type."""
        call_hash = self._compute_hash(call_type, key_inputs)
        
        # Check if we should replay
        if self.mode in ["replay", "auto"] and call_hash in self.interactions:
            if "llm" in call_type:
                self.session_metadata["llm_hits"] += 1
            else:
                self.session_metadata["tool_hits"] += 1
            
            recorded = self.interactions[call_hash]
            if self.verbose:
                print(f"🎭 REPLAY: {call_type} (hash: {call_hash[:8]}...)")
            return recorded["output"]
        
        # If replay mode but no match, error
        if self.mode == "replay":
            raise ValueError(f"Replay failed: No recorded {call_type} for hash {call_hash}")
        
        # Track miss
        if "llm" in call_type:
            self.session_metadata["llm_misses"] += 1
        else:
            self.session_metadata["tool_misses"] += 1
        
        return None  # Signal to make real call
    
    def record_generic(self, call_type: str, key_inputs: Dict[str, Any], output: Any) -> None:
        """Generic recording for any call type."""
        if self.mode in ["record", "auto"]:
            call_hash = self._compute_hash(call_type, key_inputs)
            
            # Only save if not already recorded (don't overwrite by default)
            if call_hash not in self.interactions:
                data = {
                    "type": call_type,
                    "inputs": key_inputs,
                    "output": output
                }
                self._save_interaction(call_hash, data)
                self.interactions[call_hash] = {"hash": call_hash, **data}
                if self.verbose:
                    print(f"📼 RECORD: {call_type} (hash: {call_hash[:8]}...)")
    
    def intercept_llm_with_tools(self, messages: Union[str, list], tools: list, 
                                  model: str, **kwargs) -> Optional[Any]:
        """Intercept LLM calls that include tools."""
        key_inputs = {
            "messages": self._canonical_messages(messages),
            "tools": [getattr(t, "name", str(t)) for t in tools],
            "model": model,
            **self._stable_gen_params(kwargs)
        }
        return self.intercept_generic("llm_with_tools", key_inputs)
    
    def record_llm_with_tools(self, messages: Union[str, list], tools: list,
                              model: str, output: Any, **kwargs) -> None:
        """Record LLM+tools response."""
        key_inputs = {
            "messages": self._canonical_messages(messages),
            "tools": [getattr(t, "name", str(t)) for t in tools],
            "model": model,
            **self._stable_gen_params(kwargs)
        }
        self.record_generic("llm_with_tools", key_inputs, output)
    
    def intercept_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Intercept a tool call."""
        # Canonicalize args (sort keys, stringify consistently)
        canon_args = json.loads(json.dumps(args, sort_keys=True, ensure_ascii=True))
        key_inputs = {"tool": tool_name, "args": canon_args}
        return self.intercept_generic("tool", key_inputs)
    
    def record_tool_response(self, tool_name: str, args: Dict[str, Any], 
                           result: Any) -> None:
        """Record a tool response."""
        canon_args = json.loads(json.dumps(args, sort_keys=True, ensure_ascii=True))
        key_inputs = {"tool": tool_name, "args": canon_args}
        self.record_generic("tool", key_inputs, result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        return {
            **self.session_metadata,
            "total_interactions": len(self.interactions),
            "llm_hit_rate": (
                self.session_metadata["llm_hits"] / 
                max(1, self.session_metadata["llm_hits"] + self.session_metadata["llm_misses"])
            ),
            "tool_hit_rate": (
                self.session_metadata["tool_hits"] / 
                max(1, self.session_metadata["tool_hits"] + self.session_metadata["tool_misses"])
            )
        }


class ReplayableAgent:
    """Mixin to add replay capabilities to any Agent class."""
    
    def enable_replay(self, cassette_path: str = None, mode: str = "auto", verbose: bool = False) -> 'Cassette':
        """Enable cassette-based replay for this agent."""
        self.cassette = Cassette(cassette_path, mode, verbose)
        
        # Patch the LLM's generate method
        if hasattr(self.llm, 'generate'):
            original_generate = self.llm.generate
            
            def replay_aware_generate(prompt, **kwargs):
                model = getattr(self.llm, 'model', 'unknown')
                key_inputs = {
                    "prompt": self.cassette._canonical_messages(prompt),
                    "model": model,
                    **self.cassette._stable_gen_params(kwargs)
                }
                
                replayed = self.cassette.intercept_generic("llm", key_inputs)
                if replayed is not None:
                    return replayed
                
                response = original_generate(prompt, **kwargs)
                self.cassette.record_generic("llm", key_inputs, response)
                return response
            
            self.llm.generate = replay_aware_generate
        
        # CRITICAL: Patch generate_with_tools which is what the agent actually uses
        if hasattr(self.llm, 'generate_with_tools'):
            original_gwt = self.llm.generate_with_tools
            
            def replay_aware_generate_with_tools(messages, tools, **kwargs):
                model = getattr(self.llm, 'model', 'unknown')
                replayed = self.cassette.intercept_llm_with_tools(messages, tools, model, **kwargs)
                
                if replayed is not None:
                    return replayed
                
                output = original_gwt(messages, tools, **kwargs)
                self.cassette.record_llm_with_tools(messages, tools, model, output, **kwargs)
                return output
            
            self.llm.generate_with_tools = replay_aware_generate_with_tools
        
        if verbose:
            print(f"🎬 Replay enabled: mode='{mode}', cassette='{cassette_path or self.cassette.cassette_path}'")
        return self.cassette
