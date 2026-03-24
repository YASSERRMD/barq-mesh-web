# barq-mesh-web — Public API Reference

## JavaScript / TypeScript API (wasm-bindgen)

### `BarqMeshWeb` (Phase 1+)

```ts
// Construct
const mesh = new BarqMeshWeb(collectionName: string, dim: number);

// Insert
await mesh.upsert_vector(raw_vec: number[], id: number): Promise<number>
await mesh.upsert_vectors(flat_vecs: number[], ids: number[]): Promise<number>

// Search
await mesh.search_vector(query_vec: number[], top_k: number): Promise<string> // JSON [{id,score}]

// Embedding stats (barq-wasm SIMD)
mesh.embedding_stats(raw_vec: number[]): string // JSON {mean,std_dev,norm,dim}

// Utility
mesh.count(): number
mesh.backend_info(): string
await mesh.save(): Promise<string>
await mesh.load(): Promise<string>
await mesh.clear(): Promise<void>
mesh.collection_name(): string
mesh.dim(): number
```

### `AiMesh` (Phase 3+)

```ts
const mesh = await AiMesh.create(workers: number);

await mesh.ingest(embedding: number[], metadata: any): Promise<void>
await mesh.ingest_texts(texts: string[]): Promise<number>
await mesh.retrieve(query_vec: number[], top_k: number): Promise<string>
await mesh.retrieve_hybrid(query: string, top_k: number): Promise<string>
await mesh.persist(): Promise<string>
mesh.vector_count(): number
mesh.backend(): string
await mesh.shutdown(): Promise<void>
```

### `AiMesh.run_pipeline` (Phase 4+)

```ts
// Full Plan → Execute → Verify → Critic loop
await mesh.run_pipeline(prompt: string): Promise<PipelineResult>
```

## MCP Tools (Phase 7)

| Tool | Endpoint | Request body |
|------|----------|-------------|
| `mesh.ingest_texts` | `POST /mcp/mesh.ingest_texts` | `{ texts: string[] }` |
| `mesh.retrieve_hybrid` | `POST /mcp/mesh.retrieve_hybrid` | `{ query: string, top_k: number }` |
| `mesh.retrieve` | `POST /mcp/mesh.retrieve` | `{ query: number[], top_k: number }` |
| `mesh.compute` | `POST /mcp/mesh.compute` | `{ op: string, a: number[], b: number[] }` |
| `mesh.run_pipeline` | `POST /mcp/mesh.run_pipeline` | `{ prompt: string }` |
| `mesh.stats` | `POST /mcp/mesh.stats` | `{}` |
