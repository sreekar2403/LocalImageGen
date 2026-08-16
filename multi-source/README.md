# @multi-source FastAPI Microservice

Text-to-Image/Video Generation API powered by local AI models.

## Quick Start

```bash
cd multi-source
uv pip install -e .
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

### Health Check
```bash
GET /health
```

### Image Generation (PNG/JPEG)
```bash
POST /generate/image
```

**Request Body:**
```json
{
    "prompt": "A beautiful sunset over a mountain lake",
    "fmt": "png",
    "size": 512,
    "steps": 30,
    "guidance_scale": 7.5
}
```

**Response:**
```json
{
    "success": true,
    "format": "png",
    "size_bytes": 123456,
    "data": "<base64-encoded-image>"
}
```

### Video Generation (MP4)
```bash
POST /generate/video
```

**Request Body:**
```json
{
    "prompt": "A cat walking on a grassy field",
    "duration": 30,
    "fps": 24,
    "resolution": "512x512"
}
```

### Batch Image Generation
```bash
GET /generate/image/batch?prompts=[["prompt1"],["prompt2"]]&fmt=png
```

## Usage Examples (cURL)

Generate PNG image:
```bash
curl -X POST http://localhost:8000/generate/image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat sleeping", "fmt": "png"}'
```

Generate JPEG image:
```bash
curl -X POST http://localhost:8000/generate/image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A sunset", "fmt": "jpeg"}'
```

Generate video:
```bash
curl -X POST http://localhost:8000/generate/video \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A forest scene", "duration": 15}'
```

Generate batch images:
```bash
curl "http://localhost:8000/generate/image/batch?prompts=[["cat","dog"],["sunset","mountains"]]"
```

## Supported Formats

- **Images:** PNG, JPEG (via `fmt` parameter)
- **Videos:** MP4 (via `/generate/video` endpoint)

## Installation

```bash
pip install -r multi-source/requirements.txt
pip install -r ../legacy/requirements.txt  # For model dependencies
```

## Running

```bash
uvicorn multi-source.app:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API docs at `http://localhost:8000/docs` (Swagger UI).
