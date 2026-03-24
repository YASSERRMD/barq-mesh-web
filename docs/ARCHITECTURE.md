# barq-mesh-web - Architecture

## Overview

```
barq-mesh-web
A browser-native, Rust/WASM distributed AI agent mesh
Built on: barq-wasm (SIMD compute) + barq-vweb (browser vector DB)
Pattern:  Plan → Execute → Verify → Critic agent loop
Target:   Zero server. Runs entirely in browser tabs as Web Workers.
```

## Layer Diagram

```
┌───────────────────────────────────────────────┐
│              JavaScript / Browser              │
│  BarqMeshWeb (wasm-bindgen public API)         │
└───────────────┬────────────────┬──────────────┘
                │                │
     ┌──────────▼───────┐  ┌─────▼──────────────┐
     │   barq-wasm       │  │   barq-vweb         │
     │  SIMD compute     │  │  Browser vector DB  │
     │  (Rust crate)     │  │  (npm / JS module)  │
     │                   │  │                     │
     │  vector_normalize │  │  BarqVWeb class      │
     │  cosine_similarity│  │  HNSW + BM25 + RRF  │
     │  matrix_multiply  │  │  OPFS persistence   │
     │  mean / std_dev   │  │  MiniLM-L6-v2 embed │
     └──────────────────┘  └─────────────────────┘
```

## Phases

| Phase | Module           | Adds                                     |
|-------|------------------|------------------------------------------|
| 1     | store/           | BarqVwebStore, barq-wasm normalise pipeline |
| 2     | mesh/queue, pool | WorkerPool, MeshQueue, BroadcastBus      |
| 3     | mesh/aimesh      | ingest_texts, hybrid search, OPFS persist|
| 4     | agents/          | Plan → Execute → Verify → Critic loop    |
| 5     | llm/             | WebLLM + OpenRouter + RAG context        |
| 6     | mesh/topology    | Cross-tab mesh, leader election          |
| 7     | mcp/             | MCP server via Service Worker, Dashboard |

## Data Flows

### Vector Ingest (Phase 1)
```
raw Vec<f32>
  → barq-wasm::vector_normalize()   ← SIMD L2 normalise
  → js_sys::Float32Array
  → barq-vweb::insert_vectors()     ← HNSW insert
```

### Hybrid Retrieval (Phase 3+)
```
text query
  → barq-vweb MiniLM embed          ← 384-dim
  → BM25 search  ─┐
  → HNSW kNN    ─┤ RRF merge
  → Vec<{id, score}>
```

### Agent Loop (Phase 4+)
```
PlannerAgent → ExecutorAgent → VerifierAgent
     ↑                               │ score < 0.85
     └──────────── CriticAgent ◄─────┘
```
