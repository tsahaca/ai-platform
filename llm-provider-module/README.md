# llm-provider-module

A small, zip-ready Python module that provides a common LLM interface for:
- Amazon Bedrock
- Ollama
- OpenAI-compatible chat APIs

It also includes a tiny FastAPI example.

## Start ollma server & the model 

```bash
OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" /opt/homebrew/opt/ollama/bin/ollama serve
ollama run llama3
```


## Layout

```text
llm-provider-module/
├── llm_provider/
│   ├── __init__.py
│   ├── base.py
│   ├── bedrock.py
│   ├── factory.py
│   ├── ollama.py
│   ├── openapi.py
│   └── schemas.py
├── examples/
│   └── fastapi_app.py
├── requirements.txt
└── README.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Provider selection

Set one of these:

### Bedrock

```bash
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID='anthropic.claude-3-5-sonnet-20240620-v1:0'
```

You also need valid AWS credentials with Bedrock access.

### Ollama

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL='llama3.1'
export OLLAMA_BASE_URL='http://localhost:11434'
```

Start Ollama separately and make sure the model is pulled.

### OpenAI-compatible API

```bash
export LLM_PROVIDER=openapi
export OPENAPI_MODEL='gpt-4o-mini'
export OPENAPI_BASE_URL='https://api.openai.com/v1'
export OPENAPI_API_KEY='your-key'
```

## Minimal usage

```python
from llm_provider import LLMMessage, LLMRequest, build_llm_provider

provider = build_llm_provider()
resp = provider.generate(
    LLMRequest(
        system_prompt="You are a helpful architect.",
        messages=[LLMMessage(role="user", content="Explain RAG vs fine-tuning")],
    )
)
print(resp.text)
```

## Run the FastAPI example

```bash
uvicorn examples.fastapi_app:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Generate:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "system_prompt": "You are a cloud architect assistant.",
    "messages": [
      {"role": "user", "content": "Explain tool calling vs agents."}
    ],
    "temperature": 0.2,
    "max_tokens": 200
  }'
```

## Notes

- `OpenAPIProvider` is for OpenAI-style `/chat/completions` endpoints.
- This starter focuses on text generation. Bedrock tool translation is intentionally left out to keep the module small and clear.
- For production use, add retries, structured logging, tracing, streaming, and typed tool execution.
