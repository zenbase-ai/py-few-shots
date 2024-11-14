# 🎯 FewShots: The best few shots with LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ever wished your AI model had a better memory? Meet FewShot - the simple yet powerful library for managing and retrieving few-shot examples with style! 🧠✨

## 🌟 Features

- 🚀 **Lightning Fast**: Both sync and async implementations for maximum flexibility
- 🎮 **Easy to Use**: Simple, intuitive API for managing your AI's example database
- 🔄 **Structured Output**: Support for structured outputs

## 🚀 Quick Start

```python
from sentence_transformers import SentenceTransformer # Can also use OpenAI, etc.
from few_shots.client import FewShot
from few_shots.embed.transformers import TransformersEmbedder
from few_shots.store.memory import MemoryStore

# Create a FewShot client
client = FewShot(
    embed=TransformersEmbedder(model=SentenceTransformer("all-MiniLM-L6-v2")),
    store=MemoryStore()
)

# Add some examples
client.add(
    inputs="How do I make a pizza?",
    outputs="1. Make the dough 2. Add toppings 3. Bake at 450°F"
)

# Find similar examples
results = client.list("What's the recipe for pizza?", limit=1)
for shot, similarity in results:
    print(f"Found match (similarity: {similarity:.2f}):")
    print(f"Q: {shot.inputs}")
    print(f"A: {shot.outputs}")
```

## 🔧 Installation

```bash
pip install best-shot
rye add best-shot
poetry add best-shot
```

## 🎮 Usage Examples

### Working with Structured Output I/O

```python
# Add structured data
client.add(
    inputs={"type": "greeting", "language": "English"},
    outputs={"text": "Hello, world!"}
)

# Search with similar inputs
results = client.list({"type": "greeting", "language": "English"})
```

### Async Support

```python
from few_shots.async_client import AsyncFewShot

client = AsyncFewShot(embed=async_embedder, store=async_store)

# Add examples asynchronously
await client.add(
    inputs="What's the weather like?",
    outputs="I don't have access to real-time weather data."
)

# Search asynchronously
results = await client.list("How's the weather today?", limit=1)
```

### Using LiteLLM for [Embeddings](https://docs.litellm.ai/docs/embedding/supported_embedding)

```python
from functools import partial
from litellm import aembedding
from fewshot import AsyncFewShot
from few_shots.embed.litellm import AsyncLiteLLMEmbedder

client = AsyncFewShot(
    embed=AsyncLiteLLMEmbedder(
        partial(aembedding, model="...", **kwargs),
    ),
    store=MemoryStore()
)
```

## 🛠️ Core Components

- **Shot**: The fundamental unit representing an input-output pair with a unique ID (you can use your own ID or let FewShot hash the inputs)
- **Embedder**: Converts inputs into vector embeddings for similarity search
- **Store**: Manages storage and retrieval of examples
- **Client**: Ties everything together with a clean, simple interface

## 💡 Use Cases

- 🤖 Enhance your chatbot with dynamic example retrieval
- 📚 Build a self-improving knowledge base
- 🎯 Implement context-aware few-shot learning
- 🧪 Test and experiment with different few-shot strategies

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
