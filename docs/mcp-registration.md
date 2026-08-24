# Registering LocalGen with any MCP harness

The MCP server (`app/mcp_server.py`) is a **thin stdio client**. It imports no
torch and starts in well under a second, so every harness can spawn its own copy
while a single background worker owns the GPU. If the worker is not running, the
first tool call spawns it detached and waits for `/health`.

Verified: `build_server()` takes ~680 ms and imports zero heavy modules.

## Claude Code

`.mcp.json` is already checked in at the repo root — Claude Code picks it up
automatically when you open this project. Verify with `/mcp`.

## Claude Desktop

`claude_desktop_config.json` (Windows: `%APPDATA%\Claude\`). Absolute paths are
required; Desktop does not inherit a useful working directory.

```json
{
  "mcpServers": {
    "localgen": {
      "command": "C:/Users/SREEKAR/Desktop/workspace/projects/LocalImageGen/.venv/Scripts/python.exe",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "PYTHONPATH": "C:/Users/SREEKAR/Desktop/workspace/projects/LocalImageGen"
      }
    }
  }
}
```

## OpenCode

The existing `opencode.json` entry keeps working unchanged — the module path
`app.mcp_server` did not move, only its implementation did.

```json
{
  "mcp": {
    "local-image-gen": {
      "type": "local",
      "command": ["python", "-m", "app.mcp_server"],
      "enabled": true
    }
  }
}
```

## pi / deepseek / any other stdio harness

```
command: C:/Users/SREEKAR/Desktop/workspace/projects/LocalImageGen/.venv/Scripts/python.exe
args:    ["-m", "app.mcp_server"]
env:     PYTHONPATH=C:/Users/SREEKAR/Desktop/workspace/projects/LocalImageGen
```

## Over HTTP instead of stdio

For a remote or shared harness:

```
python -m app.mcp_server --transport http
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LOCALIMAGEGEN_URL` | `http://127.0.0.1:8765` | Worker address |
| `LOCALIMAGEGEN_AUTOSPAWN` | `1` | Auto-start the worker on first tool call |
| `LOCALIMAGEGEN_LLM_PROVIDER` | `lmstudio` | `lmstudio` or `ollama` |
| `LOCALIMAGEGEN_LLM_MODEL` | `qwen3.5-4b` | Model for SVG authoring / enhancement |
| `LOCALIMAGEGEN_LMSTUDIO_URL` | `http://127.0.0.1:1234` | LM Studio server |
| `LOCALIMAGEGEN_IMAGE_MODEL` | `black-forest-labs/FLUX.2-klein-4B` | Diffusion model |
| `LOCALIMAGEGEN_ROOT` | `~/LocalImageGen` | Output root |
| `LOCALIMAGEGEN_IDLE_EVICT_S` | `600` | Idle seconds before unloading the model |

## Tools

| Tool | Returns |
|---|---|
| `generate_image` | file path + params |
| `edit_image` | file path (multi-reference editing supported) |
| `generate_svg` | SVG source inline (under 12k chars) + svg path + PNG preview path |
| `edit_svg` | as above |
| `enhance_prompt` | expanded prompt text |
| `list_presets` / `list_platforms` | platform + SVG presets |
| `service_status` / `health` | worker pid, VRAM, resident backend |
| `evict_models` | frees VRAM |

**Tools return paths, never base64.** Claude Code does not convert MCP
`ImageContent` into a native image block — base64 arrives as text at roughly
15–25k tokens per image and trips result-size limits. A path lets the harness's
own file reader render the image for ~600 tokens. SVG is the exception: it is
text, so the source is returned inline under a budget and the agent can edit it
directly.

## VRAM arbitration (important)

The GPU holds **one** model at a time (~7 GB usable of 8 GB).

- **LM Studio is GPU-accelerated and holds ~5.7 GB** while a model is resident.
  Measured: with `qwen3.5-4b` loaded, only 1155 MiB was free — not enough for
  FLUX. The manager therefore runs `lms unload --all` before any GPU lease, and
  every request carries `ttl: 300` so LM Studio also self-releases.
- **Ollama** is sent `num_gpu: 0` and `keep_alive: 0`, so it stays on the CPU and
  never contends. Switch with `LOCALIMAGEGEN_LLM_PROVIDER=ollama`.
