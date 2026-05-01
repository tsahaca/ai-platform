# Architecture

## Overview

This workspace provides a unified **LLM Provider** abstraction layer that normalises calls to multiple LLM back-ends behind a single interface, and exposes that interface through a **FastAPI** HTTP service.

---

## Package Structure

```
workspace/
├── llm_provider/          # Core library
│   ├── __init__.py        # Public API surface
│   ├── base.py            # Abstract base class
│   ├── schemas.py         # Shared dataclasses
│   ├── factory.py         # Provider factory (env-driven)
│   ├── bedrock.py         # AWS Bedrock implementation
│   ├── ollama.py          # Ollama implementation
│   └── openapi.py         # OpenAI-compatible implementation
└── examples/
    └── fastapi_app.py     # REST API example
```

---

## Class Hierarchy

```mermaid
classDiagram
    class LLMProvider {
        <<abstract>>
        +model_name: str
        +generate(request: LLMRequest) LLMResponse*
    }

    class BedrockProvider {
        +region_name: str
        +client: boto3.client
        +generate(request: LLMRequest) LLMResponse
        -_to_bedrock_message(msg: LLMMessage) dict
    }

    class OllamaProvider {
        +base_url: str
        +timeout: int
        +generate(request: LLMRequest) LLMResponse
    }

    class OpenAPIProvider {
        +base_url: str
        +api_key: str
        +timeout: int
        +generate(request: LLMRequest) LLMResponse
    }

    LLMProvider <|-- BedrockProvider
    LLMProvider <|-- OllamaProvider
    LLMProvider <|-- OpenAPIProvider
```

---

## Data / Schema Model

```mermaid
classDiagram
    class LLMMessage {
        +role: str
        +content: str
    }

    class ToolSpec {
        +name: str
        +description: str
        +input_schema: dict
    }

    class LLMRequest {
        +messages: List~LLMMessage~
        +temperature: float
        +max_tokens: int
        +system_prompt: Optional~str~
        +tools: List~ToolSpec~
        +metadata: dict
    }

    class LLMResponse {
        +text: str
        +model: str
        +provider: str
        +usage: dict
        +raw: dict
    }

    LLMRequest "1" --> "1..*" LLMMessage
    LLMRequest "1" --> "0..*" ToolSpec
```

---

## Provider Factory Flow

```mermaid
flowchart TD
    ENV["Environment Variables\nLLM_PROVIDER\nBEDROCK_MODEL_ID / OPENAPI_* / OLLAMA_*"]
    FACTORY["build_llm_provider()"]
    BEDROCK["BedrockProvider\n(AWS Bedrock Converse API)"]
    OPENAPI["OpenAPIProvider\n(OpenAI-compatible\n/chat/completions)"]
    OLLAMA["OllamaProvider\n(Ollama /api/chat)"]
    ERROR["ValueError: unsupported provider"]

    ENV --> FACTORY
    FACTORY -->|LLM_PROVIDER=bedrock| BEDROCK
    FACTORY -->|LLM_PROVIDER=openapi| OPENAPI
    FACTORY -->|LLM_PROVIDER=ollama| OLLAMA
    FACTORY -->|unknown value| ERROR
```

---

## FastAPI Request Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI (fastapi_app.py)
    participant Factory as build_llm_provider()
    participant Provider as LLMProvider
    participant Backend as LLM Backend

    Client->>FastAPI: POST /generate {messages, system_prompt, ...}
    FastAPI->>Factory: build_llm_provider()
    Factory-->>FastAPI: BedrockProvider | OllamaProvider | OpenAPIProvider
    FastAPI->>Provider: generate(LLMRequest)
    Provider->>Backend: HTTP / AWS SDK call
    Backend-->>Provider: raw response
    Provider-->>FastAPI: LLMResponse
    FastAPI-->>Client: {provider, model, text, usage}
```

---

## External Dependencies

```mermaid
graph LR
    subgraph llm_provider
        BASE[base.py]
        SCHEMAS[schemas.py]
        FACTORY[factory.py]
        BEDROCK[bedrock.py]
        OLLAMA[ollama.py]
        OPENAPI[openapi.py]
    end

    subgraph examples
        FASTAPI[fastapi_app.py]
    end

    subgraph External
        AWS[(AWS Bedrock)]
        OLLAMASRV[(Ollama Server)]
        OPENAISRV[(OpenAI-compatible API)]
    end

    FASTAPI --> FACTORY
    FACTORY --> BEDROCK
    FACTORY --> OLLAMA
    FACTORY --> OPENAPI
    BEDROCK --> BASE
    OLLAMA --> BASE
    OPENAPI --> BASE
    BASE --> SCHEMAS
    FACTORY --> BASE

    BEDROCK -->|boto3| AWS
    OLLAMA -->|HTTP requests| OLLAMASRV
    OPENAPI -->|HTTP requests| OPENAISRV
```

---

## Environment Variables Reference

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `factory.py` | Selects the active provider (`bedrock`, `openapi`, `ollama`) |
| `BEDROCK_MODEL_ID` | `BedrockProvider` | Model ID on AWS Bedrock |
| `AWS_REGION` | `BedrockProvider` | AWS region (default `us-east-1`) |
| `OPENAPI_BASE_URL` | `OpenAPIProvider` | Base URL of the OpenAI-compatible endpoint |
| `OPENAPI_API_KEY` / `OPENAI_API_KEY` | `OpenAPIProvider` | API key |
| `OPENAPI_MODEL` | `OpenAPIProvider` | Model name |
| `OLLAMA_BASE_URL` | `OllamaProvider` | Ollama server URL (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | `OllamaProvider` | Model name served by Ollama |
