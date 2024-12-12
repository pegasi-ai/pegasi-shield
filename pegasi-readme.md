# Pegasi - Metacognition for AI Agents

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Compatible-orange.svg)](https://github.com/langchain-ai/langgraph)
[![AutoGen](https://img.shields.io/badge/AutoGen-Compatible-blue.svg)](https://github.com/microsoft/autogen)
[![Crew](https://img.shields.io/badge/Crew-Compatible-green.svg)](https://github.com/crew-ai/crew)

## What is Pegasi AI? 🪽

Pegasi AI's community edition enhances AI agents with metacognitive capabilities, enabling them to reason about their own thought processes, communicate efficiently, and operate securely.  It seamlessly integrates with popular frameworks like LangGraph, AutoGen, and Crew to deploy agentic swarms and coworkers with confidence.
### Core Capabilities

#### 1. 🧠 Reasoning Enhancement: Metacognitive Knowledge Graph
- Track agent reasoning patterns and temporal memory
- Validate reasoning and capture audit trails
- Enable agent self-improvement through learning

#### 2. 🔄 Optimized Communication between AI agents
- Optimized symbolic message passing with caching
- Token-efficient protocols across frameworks
- Built for LangGraph, AutoGen, and Crew

#### 3. 🛡️ Agentic Privacy, Security, and Alignment
- Agent-to-agent authentication with RBAC
- Essential safety filters for agents
  - PII detection & filtering
  - Contradiction detection
  - Customizable input validation
- Secure embedding storage and processing 

### 🔐 Why Security Matters
Security and privacy controls are essential to protect both proprietary reasoning patterns that emerge through metacognition and sensitive user interactions that inform agent learning, ensuring your AI's intellectual property and user data remain secure as agents evolve.

### ⚡ How Pegasi Works

1. **Record & Learn**
  - Agents record reasoning in the Knowledge Graph
  - Track temporal patterns and decisions  
  - Build reusable reasoning patterns

2. **Communicate & Process**
  - Lightweight symbolic message passing
  - Built-in framework integrations
  - Efficient cross-agent protocols

3. **Secure & Validate**
  - Real-time alignment checking
  - Essential safety filtering
  - Secure agent interactions

4. **Monitor & Improve**
  - Basic performance tracking
  - Pattern recognition
  - Development insights

### ⚙️ Built for Efficiency

Pegasi is designed to be lightweight and fast:
- Smart caching of common patterns
- Minimal LLM calls for basic operations  
- Pre-computed symbolic messages
- Optimized framework integrations

### Pegasi is Efficient

The symbolic communication system is lightweight and fast. Unlike traditional approaches, Pegasi doesn't rely on heavy LLM calls for basic agent communication. It precomputes common patterns and maintains an efficient symbol cache.

Example usage:

```python
from pegasi import Agent, AlignmentToolkit

# Initialize with all three core capabilities
agent = Agent(
    # 1. Reasoning Enhancement
    metacognition_enabled=True,
    pattern_recognition=True,
    
    # 2. Optimized Communication
    symbolic_messaging=True,
    message_cache_size=1000,
    
    # 3. Security & Alignment
    alignment_toolkit=AlignmentToolkit(
        filters=["pii", "phi", "contradiction", "entailment"],
        strict_mode=True
    ),
    rbac_enabled=True
)

# Secure, efficient agent operation
result = agent.process(
    "Complex reasoning task",
    validate_reasoning=True,
    check_sensitive_data=True
)
```

## Getting Started with Agentic Frameworks

```bash
# Install Pegasi
pip install pegasi

# Optional integrations
pip install pegasi[langgraph]
pip install pegasi[autogen]
pip install pegasi[crew]

# Run example
python examples/basic_swarm.py
```

## Usage Examples

### With LangGraph
```python
from pegasi import MetaCogAgent
from langgraph.graph import Graph

# Create metacognitive agent
agent = MetaCogAgent(
    reflection_enabled=True,
    symbolic_comm=True
)

# Integrate with LangGraph
graph = Graph()
graph.add_node("metacog_agent", agent)
```

### With AutoGen
```python
from pegasi import PegasiAssistant
import autogen

# Create Pegasi-enhanced assistant
assistant = PegasiAssistant(
    name="metacog_assistant",
    metacog_enabled=True
)

# Use with AutoGen
agent = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list
    },
    system_message=assistant.system_prompt
)
```

### With Crew
```python
from pegasi import PegasiCrew
from crew import Crew

# Initialize metacognitive crew
crew = PegasiCrew(
    agents=3,
    reflection_pool=True,
    symbolic_messaging=True
)

# Run tasks with metacognition
result = crew.run("Complex task requiring reasoning")
```

## Roadmap

- [ ] Core metacognition engine
- [ ] Basic integrations
- [ ] Community features
- [ ] Additional integrations
- [ ] Extended pattern recognition
- [ ] Advanced deployment options

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Documentation

- [Installation](docs/installation.md)
- [Core Concepts](docs/concepts.md)
- [Integration Guides](docs/integrations.md)
- [API Reference](docs/api.md)

## License

Copyright 2024 Pegasi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Support

- Community Support: GitHub Issues & Discussions
- Contact: hello@pegasi.ai
