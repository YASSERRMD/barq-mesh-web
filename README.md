<div align="center">

# barq-mesh-web

<br>
<img src="assets/logo.png" alt="barq-mesh-web logo" width="400"/>
<br>

**Browser-native, Rust/WASM distributed AI agent mesh.**
<br>
Built on `barq-wasm` (SIMD compute) + `barq-vweb` (browser vector DB).

[![Build Status](https://img.shields.io/badge/build-passing-success?style=for-the-badge)](#)
[![WASM](https://img.shields.io/badge/target-wasm32--unknown--unknown-blue?style=for-the-badge&logo=webassembly)](#)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange?style=for-the-badge&logo=rust)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-magenta.svg?style=for-the-badge)](#)

</div>

---

## ⚡ Overview

**`barq-mesh-web`** is the browser-native distributed compute and execution layer for the Antigravity Agent Mesh. Using the processing horsepower of **WebAssembly (SIMD)** and **Web Workers**, it enables zero-server autonomous task routing, execution, verification, and contextual retrieval directly inside your browser tab.

It completely eliminates backend cloud dependence by bringing the entire AI Loop and Vector indexing natively to the edge.

### Key Capabilities

* **Web Worker Pooling:** Fans out dense computation across multiple isolated Web Workers utilizing `ChannelBundle` via `MessageChannel`.
* **Zero-Server RAG Engine:** Leverages `barq-vweb`'s highly optimized HNSW kNN + BM25 Hybrid engine inside WASM for robust document chunking and retrieval without an API call.
* **SIMD Accelerated Verifications:** Every generated artifact runs through a local semantic verification (cosine similarity using `barq-wasm` SIMD core math instructions) before the Agent Loop exit-gates.
* **Cross-Tab Topologies:** Uses `BroadcastChannel` for master/follower tab election resulting in safe OPFS (Origin Private File System) sync topologies.
* **MCP Desktop Extensions:** Runs an embedded Model Context Protocol (MCP) tool server locally spanning JSON-RPC for external desktop IDE mapping.

---

## 🚀 Architecture and The 7 Phases

This project was built methodically using an atomic Git commit workflow across 7 discrete architectural phases:

### Phase 1: Core Foundation & OPFS Storage 🟢
Established the core Rust scaffolding using `wasm-pack`. Integrated `barq-wasm` as a git dependency for SIMD routines and created the `BarqMeshWeb` facade for connecting `barq-vweb` Javascript dependencies natively into Rust.

### Phase 2: Mesh Queue & Worker Pool 🟢
Created `MeshQueue` offering dynamic deduplication using semantic verification alongside a massively parallel `WorkerPool` using cross-thread `MessageChannel`s targeting worker isolates.

### Phase 3: Text Ingestion & Hybrid Search 🟢
Bound high-level text ingestion directly to `barq-vweb` ONNX-MiniLM-v6 transformers. `retrieve_hybrid()` maps robust keyword and dense matching algorithm outputs mapped directly through Rust WASM boundaries. 

### Phase 4: Antigravity Agent Loop 🟢
Orchestrated the core AI execution loop locally: `Planner` → `Executor` → `Verifier` → `Critic`. Implemented SIMD semantic validation (target >0.85 evaluation gate threshold). 

### Phase 5: LLM Router & Native Integrations 🟢
Built `LlmRouter` enabling seamless dynamic prompt RAG-augmentation mapping to native browser offline deployments (WebLLM/Phi-3/Qwen2.5) gracefully falling back to Cloud API (OpenRouter).

### Phase 6: Cross-Tab Mesh Distributed Computing 🟢
Embedded `TopologyManager`. Maps tabs as distributed peer-to-peer compute nodes, electing singleton 'Leader' tabs via native Broadcast protocols reducing race conditions against browser OPFS persistence engines.

### Phase 7: MCP Server Native Mode 🟢
Configured `McpServer`; parses, coordinates, and routes strict JSON-RPC payload requests. Exposes local web instances to desktop TCP proxy routers enabling cursor/vscode zero-friction RAG deployment.

---

## 💻 Running the Application

`barq-mesh-web` is built using standard Rust `wasm-pack` build tools.

```bash
# Compile and build the target library for WASM
$ cargo build --target wasm32-unknown-unknown --release --features wasm
```

### Examples

The system components can be natively evaluated locally via `index.html` UX dashboards located iteratively in the `examples/` directory:

1. `examples/phase1_store/index.html` (Basic HNSW embedding indexing)
2. `examples/phase2_mesh/index.html` (Web Worker queue routing validation)
3. `examples/phase3_search/index.html` (RAG vector hybrid searching matrix)
4. `examples/phase4_agents/index.html` (Agent Loop Validation threshold simulation)
5. `examples/phase5_llm/index.html` (LlmRouter prompt injections mapping via AiMesh context)
6. `examples/phase6_crosstab/index.html` (Multi-tab cluster topology election logic)
7. `examples/phase7_mcp/index.html` (JSON-RPC MCP native browser tool integration loop)

Serve the directory utilizing a standard web server bridging appropriate browser headers *(Requires `Cross-Origin-Opener-Policy: same-origin` restrictions enabled to support explicit SharedArrayBuffers)*.

---

<div align="center">
<b>barq-mesh-web</b> • Antigravity Advanced Agentic Systems • 2026
</div>
