# barq-mesh-web - Phases

## Phase 1 - Core Foundation
**Status:**  Implemented

Scaffolds the project. Wires `barq-wasm` and `barq-vweb` together.
Basic vector insert and kNN search in the browser.

**Exit gate:**
- `wasm-pack build --target web --features wasm` → SUCCESS
- Insert 100 × 384-dim vectors; kNN top-5 < 5 ms
- `backend_info()` returns SIMD tier string

---

## Phase 2 - Mesh Queue + Worker Pool
**Status:** ⏳ Planned

Rust spawns N Web Workers. `MeshQueue` fans tasks out by priority.
Semantic dedup via barq-vweb + barq-wasm rejects near-duplicates.

**Exit gate:**
- Submit 50 tasks → all 50 `AiMeshResult` returned
- ~3–5 near-duplicate Ingest tasks dropped by dedup
- `pool.stats()` → `{ workers:4, busy:0, queued:0, completed:50 }`

---

## Phase 3 - Text Ingestion + Hybrid Search
**Status:** ⏳ Planned

`insert_texts` → MiniLM-L6-v2 → HNSW + BM25.
`retrieve_hybrid` → BM25 + HNSW + RRF merged.
OPFS save/load persists across reloads.

**Exit gate:**
- `ingest_texts(500 chunks)` → `vector_count() === 500`
- `retrieve_hybrid("query", 5)` → `r[0].score > 0.70` in < 10 ms
- Reload page → OPFS restores `vector_count() === 500`

---

## Phase 4 - Antigravity Agent Loop
**Status:** ⏳ Planned

Full **Plan → Execute → Verify → Critic** loop in Rust.
Every agent action produces a typed `Artifact` stored in barq-vweb.
`VerifierAgent` uses `cosine_similarity_simd` as proof-of-work.

**Exit gate:**
- `run_pipeline("build a Rust WASM demo")` → 3 steps planned
- VerifierAgent score ≥ 0.85 → `ProofArtifact { passed: true }`
- Retry triggers when score < 0.85

---

## Phase 5 - LLM Integration
**Status:** ⏳ Planned

WebLLM (offline: Phi-3/Qwen2.5) + OpenRouter (online).
barq-vweb provides RAG context for every prompt.
barq-wasm verifies every LLM output semantically.

**Exit gate:**
- `run_pipeline("explain HNSW")` complete in < 30 s (WebLLM cold start)
- VerifierAgent cosine score ≥ 0.85

---

## Phase 6 - Cross-Tab Mesh + OPFS Persistence
**Status:** ⏳ Planned

Multiple browser tabs form a distributed mesh via `BroadcastChannel`.
All tabs share the same HNSW index state through OPFS.

**Exit gate:**
- Tab 1 ingests 1000 chunks, saves; Tab 2 restores → same results
- BroadcastChannel shows Tab 2 received Tab 1's artifacts

---

## Phase 7 - MCP Server + Dashboard
**Status:** ⏳ Planned

Service Worker exposes MCP JSON-RPC on `/mcp/*`.
Dashboard: worker states, vector count, agent loop, barq-wasm metrics.

**Exit gate:**
- `POST /mcp/mesh.retrieve_hybrid` → results in < 15 ms
- All 6 MCP tools functional

---

## Release Milestones

| Milestone | Phases | Tag |
|-----------|--------|-----|
| Alpha | 1 + 2 | `v0.1.0-alpha` |
| Beta  | 3 + 4 | `v0.2.0-beta`  |
| RC    | 5 + 6 | `v0.3.0-rc`    |
| v1.0  | 7     | `v1.0.0`       |
