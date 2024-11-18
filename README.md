# 🎯 FewShots: The best few shots with LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ever wished your AI model had a better memory? Meet `few-shots` - the simple yet powerful library for managing and retrieving few-shot examples with style! 🧠✨

## 🌟 Features

- 🎮 **Easy to Use**: Simple, intuitive API for managing your AI's example database
- 🔄 **Structured Output**: Support for structured outputs

## 💡 Use Cases

- 🤖 Enhance your chatbot with dynamic example retrieval
- 📚 Build a self-improving knowledge base
- 🎯 Implement context-aware few-shot learning

## 🛠️ Core Components

- **Shot**: The fundamental unit representing an input-output pair with a unique ID (bring your own ID or let FewShots hash the inputs)
- **Embed**: Converts inputs into vector embeddings for similarity search
- **Store**: Manages storage and retrieval of examples
- **Client**: Ties everything together with a clean, simple interface

## 🔧 Installation

```bash
pip install few-shots
rye add few-shots
poetry add few-shots
```

## 🚀 Quick Start

```python
from few_shots.client import FewShots
from few_shots.embed.openai import OopenAIEmbed
from few_shots.store.memory import MemoryStore # see below for different vectorstores
from openai import OpenAI

shots = FewShots(
    embed=OpenAIEmbed(
        OpenAI().embeddings.create,
        model="...",
        **kwargs,
    ),
    store=MemoryStore()
)

shots.add(
    inputs="How do I make a pizza?",
    outputs="1. Make the dough 2. Add toppings 3. Bake at 450°F"
)

# Or use structured inputs/outputs
shots.add(
    inputs=dict,
    outputs=dict,
)

# Use your own foreign key for upserts
shots.add(
    inputs=...,
    outputs=...,
    id=str,
)

# Find similar examples
best_shots = shots.list(
    "What's the recipe for pizza?",
    limit=10, # default = 5
)
for distance, shot in results:
    print(f"Found match (distance: {distance:.2f}):")
    print(f"Q: {shot.inputs}")
    print(f"A: {shot.outputs}")

# Use with your LLM
from few_shots.utils.format import shots_to_messages

openai.chat.completions.create(
    ...,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        *shots_to_messages(best_shots),
        {"role": "user", "content": "What's the recipe for pizza?"},
    ]
)
```


## 🎮 Usage Examples

### Working with Structured Output I/O

```python
# Add structured data
shots.add(
    inputs={"type": "greeting", "language": "English"},
    outputs={"text": "Hello, world!"}
)

# Search with similar inputs
best_shots = shots.list({"type": "greeting", "language": "English"})
```

### Using persistent Vector Stores

```python
from few_shots.store.pg import PGStore, AsyncPGStore
from few_shots.store.chroma import ChromaStore, AsyncChromaStore
from few_shots.store.qdrant import QdrantStore, AsyncQdrantStore
from few_shots.store.weaviate import WeaviateStore, AsyncWeaviateStore
from few_shots.store.turbopuffer import TurboPufferStore # Untested
from few_shots.store.milvus import MilvusStore # TODO

# check out the store's .setup method to see how to configure it
# this method creates the table, collection, indexes, etc. and is idempotent
```

### Using OpenAI / LiteLLM for [Embeddings](https://docs.litellm.ai/docs/embedding/supported_embedding)

The `OpenAIEmbed` and `AsyncOpenAIEmbed` classes are compatible with all OpenAI-compatible SDKs.

```python
from few_shots import AsyncFewShots
from few_shots.embed.openai import OpenAIEmbed, AsyncOpenAIEmbed # Compatible with all OpenAI

from openai import OpenAI

shots = FewShots(
    embed=OpenAIEmbed(
        OpenAI().embeddings.create,
        model="...",
        **kwargs,
    ),
    store=MemoryStore()
)

from litellm import aembedding

shots = AsyncFewShots(
    embed=AsyncOpenAIEmbed(
        aembedding,
        model="...",
        **kwargs,
    ),
    store=MemoryStore()
)
```

## 🤝 Contributing

We love contributions! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

MIT License - feel free to use it in your projects!

---

Made with ❤️ by developers who believe in the power of learning from examples.

*Remember: The best AI is the one that learns from experience!* 🌟
