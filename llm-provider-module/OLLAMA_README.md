You usually **do not need to run both manually**.

They do different things:

* `ollama serve` starts the Ollama API server, which serves requests on `http://localhost:11434/api`. That is the process your app or `curl` talks to. ([Ollama Documentation][1])
* `ollama run llama3` runs a model for interactive use. If the model is not already present, it will pull it first, then start a chat session with that model. The Llama 3 library page shows `ollama run llama3` as the standard CLI way to start using the model. ([Ollama Documentation][2])

So in practice:

## When you need only `ollama serve`

Use this when your **FastAPI app, devcontainer, or any client code** will call Ollama over HTTP.

Example:

```bash
OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" /opt/homebrew/opt/ollama/bin/ollama serve
```

That starts the server with those environment settings applied to the server process. The docs also note that environment variables are set on the Ollama service/process itself. ([Ollama Documentation][3])

Then your app can call:

```text
http://localhost:11434/api/chat
```

or

```text
http://localhost:11434/api/generate
```

without opening the CLI chat. ([Ollama Documentation][1])

## When you need only `ollama run llama3`

Use this when you just want to:

* pull the model if needed
* test it interactively in the terminal

Example:

```bash
ollama run llama3
```

This is useful for a quick sanity check, but it is not required for your app if the server is already running and the model has been pulled. ([Ollama][4])

## Why people sometimes run both

Because the first command starts the server with custom performance settings, and the second command is a convenient way to make sure `llama3` is downloaded and works.

Typical sequence:

1. Start server with tuning flags:

   ```bash
   OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve
   ```
2. In another terminal, pull/test model once:

   ```bash
   ollama run llama3
   ```

On Linux docs, Ollama explicitly shows `ollama serve` in one terminal and then using another terminal to interact/verify. ([Ollama Documentation][5])

## For your setup

Since you want Ollama outside the devcontainer and your app inside the devcontainer, the better pattern is:

1. On host:

   ```bash
   OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve
   ```
2. Pull the model once, if needed:

   ```bash
   ollama pull llama3
   ```

   or use `ollama run llama3` once just to download/test it.
3. In your app, set:

   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   OLLAMA_MODEL=llama3
   ```

After that, your FastAPI app can call Ollama directly. You do **not** need to keep running `ollama run llama3` every time.

## Best simple rule

* **For app/API use:** run `ollama serve`
* **For terminal chat/manual test:** run `ollama run llama3`

A cleaner setup for you is:

```bash
OLLAMA_FLASH_ATTENTION="1" OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve
```

and then once:

```bash
ollama pull llama3
```

That avoids opening the interactive CLI every time.

[1]: https://docs.ollama.com/api/introduction?utm_source=chatgpt.com "Introduction"
[2]: https://docs.ollama.com/cli?utm_source=chatgpt.com "CLI Reference"
[3]: https://docs.ollama.com/faq?utm_source=chatgpt.com "FAQ"
[4]: https://ollama.com/library/llama3?utm_source=chatgpt.com "llama3"
[5]: https://docs.ollama.com/linux?utm_source=chatgpt.com "Linux"
