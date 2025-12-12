# Lab 05: Building AI Agents

Build **AI agents** that reason, use tools, and complete tasks autonomously - all locally.

## Prerequisites

```bash
pip install langchain langchain-community ollama
ollama pull llama3.2
```

## What You'll Learn

- Creating custom tools
- Building ReAct agents (Reasoning + Acting)
- Multi-step task execution
- Customer service agent example
- Simple agents without frameworks

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_building_agents.ipynb` | Complete agent building guide |

## Agent Architecture

```
User Input → Agent (LLM) → Thought → Action → Tool → Observation → Repeat → Final Answer
```

## Built-in Tools

| Tool | Purpose |
|------|---------|
| calculator | Math operations |
| search_knowledge_base | Policy lookups |
| lookup_order | Order status |
| initiate_return | Process returns |
| escalate_to_human | Hand off complex issues |

## ReAct Pattern

```
Question: What's the status of order ORD-001?
Thought: I need to look up this order
Action: lookup_order
Action Input: ORD-001
Observation: {"status": "shipped", "tracking": "..."}
Thought: I have the information
Final Answer: Your order has shipped!
```

## Time to Complete

Approximately 30 minutes
