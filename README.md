# ollama-rdna1

By [artium-projects.com](https://artium-projects.com) -- a step-by-step guide on how to create and deploy this project.

> Tried running local LLMs on your RX 5700 XT? Discovered AMD dropped ROCm support? Searched everywhere and found no solution? So did I -- this is the solution I wish I'd found.

A drop-in replacement for [Ollama](https://ollama.com) for AMD GPUs that lost ROCm support -- specifically RDNA 1 (gfx1010) cards like the RX 5500 XT, RX 5600 XT, and RX 5700 XT.

Uses [llama.cpp](https://github.com/ggml-org/llama.cpp) with a Vulkan backend instead of ROCm. No ROCm installation required. Provides Ollama-compatible and OpenAI-compatible APIs on port 11434, pulls models from both the Ollama registry and HuggingFace, and supports text-only and multimodal (vision + audio) models.

**Linux only.** This problem is Linux-specific -- ROCm only exists on Linux. On Windows, Ollama uses DirectML or CUDA instead of ROCm, so RDNA 1 GPUs are not affected.

---

## Why This Exists

### The GPU

AMD GPUs have codenames based on their architecture:

```
gfx1010  =  RDNA 1  =  RX 5500/5600/5700 series
gfx1030  =  RDNA 2  =  RX 6000 series
gfx1100  =  RDNA 3  =  RX 7000 series
```

### ROCm Dropped RDNA 1

ROCm is AMD's GPU compute stack (their equivalent of NVIDIA's CUDA). It lets software run math on the GPU for AI workloads.

In ROCm 6.x, AMD **removed support for RDNA 1 (gfx1010)**. Only RDNA 2 and newer are supported. The GPU still works for gaming and display -- only the compute/AI side is dropped.

### Why Ollama Fails

Ollama uses ROCm to run models on AMD GPUs. Here is what happens at startup:

```
Ollama starts
    |
Reads /sys/class/kfd/kfd/topology/        "I see an AMD GPU"
    |
Loads its BUNDLED ROCm libraries           (not your system ones)
    |
Checks librocblas.so for precompiled       looks for .co kernel files
kernels for your GPU architecture          matching gfx1010
    |
Finds ZERO kernels for gfx1010            "this GPU is not supported"
    |
Reports: total vram = 0 B                 falls back to CPU only
```

The critical detail: Ollama **ships its own ROCm** inside `/usr/local/lib/ollama/rocm/`. Even if you install system ROCm with gfx1010 hacks, Ollama ignores it and uses its bundled copy, which has no gfx1010 support.

Common workarounds that **do not work**:

- `HSA_OVERRIDE_GFX_VERSION=10.3.0` -- the filter runs before HSA is involved.
- `OLLAMA_VULKAN=1` -- the official Ollama binary is **not compiled with Vulkan support**. There is no Vulkan code path in the binary.

References: [ollama/ollama#8806](https://github.com/ollama/ollama/issues/8806), [ollama/ollama#2503](https://github.com/ollama/ollama/issues/2503)

### What Ollama Actually Is

Ollama is a wrapper. Under the hood:

```
Ollama  =  model management  +  API server  +  CLI
                |
           llama.cpp  =  the actual inference engine that loads
                          GGUF model files and runs tensor math on the GPU
```

Ollama bundles [llama.cpp](https://github.com/ggml-org/llama.cpp) inside itself. llama.cpp is the open-source project that does the real work. Ollama's bundled copy is compiled with ROCm only.

### The Fix: Build llama.cpp with Vulkan

Since Ollama's bundled llama.cpp only supports ROCm, ollama-rdna1 builds llama.cpp from source with Vulkan instead:

```
What Ollama does:                     What ollama-rdna1 does:

Ollama binary                         ollama-rdna1 (bash script)
  +-- bundled llama.cpp                 +-- llama.cpp built from source
        +-- ROCm backend                      +-- Vulkan backend
              +-- gfx1010 NOT supported              +-- works on ANY GPU with Vulkan
```

The cmake flag that makes this work:

```bash
cmake -B build -DGGML_VULKAN=ON
```

Vulkan does not care about AMD architecture codenames. It talks to the [Mesa RADV](https://docs.mesa3d.org/drivers/radv.html) driver, which fully supports RDNA 1. No ROCm involved at all.

### What ollama-rdna1 Adds

llama.cpp has a server and CLI but no Ollama-compatible API. Clients that expect the Ollama API (Open WebUI, Continue, etc.) will not work directly. ollama-rdna1 bridges this gap:

```
Application (Open WebUI, Continue, Python, curl)
    |
    calls http://localhost:11434/api/chat    (Ollama API)
    |
ollama-rdna1 proxy  (ollama_proxy.py)
    |
    translates to http://localhost:8081/v1/chat/completions
    |
llama-server  (llama.cpp, Vulkan backend)
    |
    GPU via:  Vulkan -> Mesa RADV -> your AMD GPU
```

Same models, same API, same port as Ollama -- just Vulkan instead of ROCm.

### The Multimodal Complication

Multimodal models (like Gemma 4 with vision and audio) have an additional complication. The Ollama registry bundles all model components into a single GGUF file:

```
Ollama registry format:                 What llama.cpp expects:

One file, ~9 GB                         Two separate files:
  +-- 720 text tensors                    +-- text.gguf (720 text tensors)
  +-- ~800 vision tensors                 +-- mmproj.gguf (vision + audio)
  +-- ~600 audio tensors
  = 2131 total
```

llama.cpp's model loader requires every tensor in a file to be consumed. When it opens the single Ollama file and only builds the text model (720 tensors), it crashes because the file header declared 2131.

The solution: pull multimodal models from HuggingFace, where [ggml-org](https://huggingface.co/ggml-org) publishes properly split files. ollama-rdna1 handles this with the `hf:` prefix:

```bash
ollama-rdna1 pull hf:ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M
```

This downloads both the text model and the multimodal projector as separate files. ollama-rdna1 auto-detects the projector and uses the right binary.

### The Full Stack

```
+------------------------------------------------------------+
|  Application (Open WebUI, Continue, Python, curl)          |
+------------------------------------------------------------+
|  ollama-rdna1                                               |
|    +-- ollama_proxy.py    API format translation            |
|    +-- model management   pull from Ollama + HuggingFace    |
|    +-- multimodal detect  auto-finds mmproj files           |
+------------------------------------------------------------+
|  llama.cpp (built from source with -DGGML_VULKAN=ON)       |
|    +-- llama-server       HTTP API on port 8081             |
|    +-- llama-cli          text-only interactive chat        |
|    +-- llama-mtmd-cli     multimodal interactive chat       |
+------------------------------------------------------------+
|  Vulkan API                                                |
+------------------------------------------------------------+
|  Mesa RADV driver                                          |
+------------------------------------------------------------+
|  AMD GPU (RDNA 1 / gfx1010 or any GPU with Vulkan)        |
+------------------------------------------------------------+
```

---

## Tested Hardware

| GPU | Architecture | VRAM | Status |
|---|---|---|---|
| AMD Radeon RX 5700 XT | RDNA 1 (gfx1010) | 8 GB | Confirmed working |

**Should also work with:** RX 5500 XT, RX 5600 XT, RX 5700, and potentially older GCN GPUs -- any AMD GPU with working Vulkan support via Mesa RADV.

**Not limited to AMD.** The Vulkan backend works with any GPU that has Vulkan drivers, including NVIDIA and Intel. However, NVIDIA users are better served by Ollama's native CUDA support, and this project is specifically motivated by the ROCm gap.

### Performance (RX 5700 XT)

| Model | Type | Quant | Prompt | Generation |
|---|---|---|---|---|
| Qwen 2.5 1.5B | text | Q4\_K\_M | ~260 t/s | ~128 t/s |
| Gemma 4 E4B | multimodal | Q4\_K\_M | ~141 t/s | ~55 t/s |

---

## Prerequisites

```bash
sudo apt install cmake build-essential python3 curl \
    libvulkan-dev vulkan-tools glslang-tools glslc spirv-headers
```

Verify your GPU is visible to Vulkan:

```bash
vulkaninfo --summary 2>/dev/null | grep -E "GPU|driverName|apiVersion"
```

You should see `driverName = radv` and `apiVersion` 1.2 or higher. If nothing appears, install `mesa-vulkan-drivers`.

---

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/myusername68/ollama-rdna1 ~/ollama-rdna1
cd ~/ollama-rdna1
```

### 2. Clone and build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp

cmake -B build \
    -DGGML_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

Build time is roughly 5-10 minutes.

### 3. Verify Vulkan is detected

```bash
./build/bin/llama-cli --list-devices
```

Expected output (your GPU name will vary):

```
Available devices:
  Vulkan0: AMD Radeon RX 5700 (RADV NAVI10) (8192 MiB, 7290 MiB free)
```

### 4. Add to PATH

```bash
cd ..
chmod +x ollama-rdna1
ln -sf "$(pwd)/ollama-rdna1" ~/.local/bin/ollama-rdna1
```

---

## Quick Start

### Text-only model

```bash
# Pull a model from the Ollama registry
ollama-rdna1 pull qwen2.5:1.5b

# Interactive chat
ollama-rdna1 run qwen2.5:1.5b

# Or start the API server
ollama-rdna1 serve qwen2.5:1.5b
```

### Multimodal model (vision + audio)

```bash
# Pull from HuggingFace (auto-downloads multimodal projector)
ollama-rdna1 pull hf:ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M

# Interactive chat (auto-detects multimodal)
ollama-rdna1 run gemma4:e4b-q4km

# In the chat, use /image or /audio commands:
#   /image photo.jpg
#   /audio recording.wav
#   What do you see?
```

### API server

```bash
ollama-rdna1 serve gemma4:e4b-q4km

# Test it
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

---

## Commands

| Command | Description |
|---|---|
| `ollama-rdna1 serve [model]` | Start API server on port 11434. Uses first available model if none specified. |
| `ollama-rdna1 run <model>` | Interactive chat. Auto-detects multimodal models. |
| `ollama-rdna1 list` | List downloaded models with sizes and `[multimodal]` tags. |
| `ollama-rdna1 pull <model>` | Pull a model (see sources below). |
| `ollama-rdna1 stop` | Stop the running server. |
| `ollama-rdna1 ps` | Show server status, PID, and loaded model. |

### Pull sources

```bash
# Ollama registry (text-only models)
ollama-rdna1 pull qwen2.5:1.5b
ollama-rdna1 pull llama3.2
ollama-rdna1 pull mistral:7b

# HuggingFace (multimodal or text, auto-downloads mmproj if available)
ollama-rdna1 pull hf:ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M
ollama-rdna1 pull hf:ggml-org/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M

# Direct URL
ollama-rdna1 pull https://huggingface.co/user/repo/resolve/main/model.gguf
```

> **Note:** Multimodal models from the Ollama registry are bundled as a single GGUF file, which llama.cpp cannot load (see [Why This Exists](#the-multimodal-complication)). Use the `hf:` prefix to pull from HuggingFace where they are properly split.

---

## API

The server listens on `http://0.0.0.0:11434` (same default port as Ollama).

| Endpoint | Method | Format | Description |
|---|---|---|---|
| `/api/chat` | POST | Ollama | Chat completions |
| `/api/generate` | POST | Ollama | Text completions |
| `/api/tags` | GET | Ollama | List available models |
| `/api/show` | POST | Ollama | Model info |
| `/v1/chat/completions` | POST | OpenAI | Chat completions |
| `/` | GET | -- | Health check |

### Ollama format

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:1.5b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

### OpenAI format

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Vision (image input)

```bash
IMG_B64=$(base64 -w0 photo.png)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG_B64"'"}}
      ]
    }]
  }'
```

### Python (OpenAI SDK)

Any client that supports a custom OpenAI base URL will work:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="none")

response = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### Python (vision)

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="none")

with open("photo.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gemma4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]
)
print(response.choices[0].message.content)
```

---

## Multimodal Models

ollama-rdna1 auto-detects multimodal models by looking for an `mmproj-*.gguf` file alongside the main model GGUF. When found:

- `run` uses `llama-mtmd-cli` instead of `llama-cli`
- `serve` passes `--mmproj` to `llama-server`
- `list` shows `[multimodal]` next to the model name

A multimodal model like Gemma 4 has three components:

| Component | File | Purpose |
|---|---|---|
| Text model | `gemma-4-E4B-it-Q4_K_M.gguf` | Processes text tokens |
| Vision encoder | `mmproj-gemma-4-E4B-it-Q8_0.gguf` | Converts images to token embeddings |
| Audio encoder | (same mmproj file) | Converts audio to token embeddings |

The vision and audio encoders produce embeddings in the same vector space as text tokens, so the text model can process them together.

---

## What Fits in 8 GB VRAM

| Model | Type | Quant | VRAM | Pull command |
|---|---|---|---|---|
| Qwen 2.5 1.5B | text | Q4\_K\_M | ~1.2 GB | `ollama-rdna1 pull qwen2.5:1.5b` |
| Llama 3.2 3B | text | Q4\_K\_M | ~2.4 GB | `ollama-rdna1 pull llama3.2` |
| Gemma 4 E4B | multimodal | Q4\_K\_M | ~3.5 GB | `ollama-rdna1 pull hf:ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` |
| Mistral 7B | text | Q4\_K\_M | ~4.8 GB | `ollama-rdna1 pull mistral:7b` |
| Qwen 2.5 7B | text | Q4\_K\_M | ~5.0 GB | `ollama-rdna1 pull qwen2.5:7b` |

Models up to ~7B parameters at Q4 quantization fit comfortably, leaving headroom for context.

---

## Architecture

```
                    +-----------------+
  Client (:11434)   |  ollama_proxy   |   Translates Ollama API <-> llama.cpp
  ----requests----> |   (Python)      |
                    +--------+--------+
                             |
                             v
                    +--------+--------+
                    |  llama-server   |   llama.cpp HTTP server
                    |    (:8081)      |   Vulkan GPU backend
                    +-----------------+
```

**`ollama-rdna1`** (Bash) -- CLI entry point. Manages model resolution, server lifecycle, and model downloads from the Ollama registry and HuggingFace.

**`ollama_proxy.py`** (Python, zero dependencies) -- Reverse proxy that translates between Ollama and llama.cpp API formats:
- `/api/chat` -> `/v1/chat/completions` (response rewritten to Ollama JSON)
- `/api/generate` -> `/completion` (response rewritten to Ollama JSON)
- `/v1/*` -> passthrough (native OpenAI-compatible endpoints)
- Streaming uses Ollama-style NDJSON with `done: false/true`

**`llama-server`** (C++, from llama.cpp) -- Inference engine on backend port 8081. Handles model loading, GPU offloading via Vulkan, text generation, and multimodal input (images, audio) when `--mmproj` is provided.

### Project layout

```
ollama-rdna1/
    ollama-rdna1          # CLI entry point (bash)
    ollama_proxy.py      # API translation proxy (python, zero deps)
    llama.cpp/           # llama.cpp (cloned separately, not included)
        build/bin/
            llama-server
            llama-cli
            llama-mtmd-cli
    models/              # downloaded GGUF files
        .manifests/      # model name -> filename mapping (JSON)
```

---

## Configuration

Edit the top of the `ollama-rdna1` script:

```bash
PORT=11434          # Ollama API port (default: same as Ollama)
BACKEND_PORT=8081   # llama-server internal port
GPU_LAYERS=99       # layers to offload to GPU (99 = all)
```

`GPU_LAYERS=99` offloads all layers to the GPU. Reduce this if a model exceeds your VRAM -- layers not offloaded fall back to CPU.

---

## Troubleshooting

**`vulkaninfo` shows no devices**

```bash
sudo apt install mesa-vulkan-drivers
ls /usr/share/vulkan/icd.d/radeon_icd*.json
```

**Build fails with `glslc not found`**

```bash
sudo apt install glslang-tools glslc
```

**Build fails with `spirv/unified1/spirv.hpp: No such file`**

```bash
sudo apt install spirv-headers
```

If you cannot install system packages:

```bash
mkdir -p ~/.local/include/spirv/unified1
curl -sL https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/main/include/spirv/unified1/spirv.hpp \
    -o ~/.local/include/spirv/unified1/spirv.hpp

cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-I$HOME/.local/include" \
    -DCMAKE_C_FLAGS="-I$HOME/.local/include"
```

**Model runs at CPU speed**

Check `/tmp/ollama-rdna1.log` for `ggml_vulkan: Using...`. If absent, Vulkan was not detected. Verify with `vulkaninfo --summary`.

**"Unknown model" error**

```bash
ollama-rdna1 list
ollama-rdna1 pull qwen2.5:1.5b
```

**Multimodal model fails with "wrong number of tensors"**

The model was pulled from the Ollama registry as a single file. Pull from HuggingFace instead:

```bash
ollama-rdna1 pull hf:ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M
```

**Port 11434 already in use**

```bash
sudo systemctl stop ollama    # stop the real Ollama if running
ollama-rdna1 serve
```

**"custom template is not supported"**

Add `--jinja` when calling llama-server or llama-mtmd-cli directly. ollama-rdna1 does this automatically for multimodal models.

---

## Contributing

Issues and pull requests are welcome. If you have tested ollama-rdna1 on a GPU not listed above, please open an issue with your hardware details and performance numbers.

---

## License

MIT
