//! barq_vweb_store — Phase 1 core
//!
//! Wraps the `barq-vweb` JS class (`BarqVWeb`) via wasm-bindgen extern blocks,
//! and exposes a clean Rust/WASM public API as `BarqMeshWeb`.
//!
//! barq-vweb JS API consumed here (from https://github.com/YASSERRMD/barq-vweb):
//!   new BarqVWeb(collection_name, null)
//!   insert_vectors(Float32Array, Uint32Array, dim) → Promise<number>
//!   search_vector(Float32Array, top_k)             → Promise<SearchResult[]>
//!   count()                                         → number
//!   backend_info()                                  → string
//!   save()                                          → Promise<string>
//!   load()                                          → Promise<string>
//!   clear()                                         → Promise<bool>
//!
//! barq-wasm Rust API used here (from https://github.com/YASSERRMD/barq-wasm):
//!   vector_normalize(&[f32]) → Vec<f32>
//!   vector_norm_simd(&[f32]) → f32
//!   mean(&[f32])             → f32
//!   std_dev(&[f32])          → f32

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen_futures::JsFuture;
#[cfg(feature = "wasm")]
use js_sys::{Float32Array, Uint32Array, Promise, Reflect, Array};

use crate::mesh::types::{SearchResult, EmbeddingStats};

// ── barq-wasm Rust functions (direct Rust calls, zero JS overhead) ─────────
#[cfg(feature = "wasm")]
use barq_wasm::wasm_bindings::{
    vector_normalize,
    vector_norm_simd,
    mean,
    std_dev,
};

// ── barq-vweb JS extern block ──────────────────────────────────────────────
//
// We declare the minimal surface of BarqVWeb that Phase 1 needs.
// barq-vweb is loaded by the page as a JS module; wasm-bindgen routes
// these calls through the JS/WASM boundary at zero extra overhead.
//
// NOTE: The JS class is named `BarqVWeb` in the barq-vweb npm package.
//       We bind it here under the Rust name `BarqVWebJs` to avoid
//       clashing with our public `BarqMeshWeb` struct.

#[cfg(feature = "wasm")]
#[wasm_bindgen(module = "barq-vweb")]
extern "C" {
    /// `BarqVWeb` from https://github.com/YASSERRMD/barq-vweb
    #[wasm_bindgen(js_name = "BarqVWeb")]
    type BarqVWebJs;

    /// `new BarqVWeb(collection_name, model_url?)`
    #[wasm_bindgen(constructor, js_class = "BarqVWeb")]
    fn new(collection_name: &str, model_url: JsValue) -> BarqVWebJs;

    /// Insert pre-computed float vectors.
    /// `insert_vectors(vectors: Float32Array, ids: Uint32Array, dim: number) → Promise<number>`
    #[wasm_bindgen(method, js_name = "insert_vectors")]
    fn insert_vectors(
        this: &BarqVWebJs,
        vectors: &Float32Array,
        ids: &Uint32Array,
        dim: usize,
    ) -> Promise;

    /// kNN vector search.
    /// `search_vector(query: Float32Array, top_k: number) → Promise<{id,score}[]>`
    #[wasm_bindgen(method, js_name = "search_vector")]
    fn search_vector(
        this: &BarqVWebJs,
        query: &Float32Array,
        top_k: usize,
    ) -> Promise;

    /// Insert texts.
    #[wasm_bindgen(method, js_name = "insert_texts")]
    fn insert_texts(this: &BarqVWebJs, texts: &Array, metadata: &Array) -> Promise;

    /// Search hybrid.
    #[wasm_bindgen(method, js_name = "search")]
    fn search(this: &BarqVWebJs, query: String, top_k: usize, hybrid: bool) -> Promise;

    /// Returns the number of indexed vectors.
    #[wasm_bindgen(method, js_name = "count")]
    fn count(this: &BarqVWebJs) -> usize;

    /// Returns a string describing the active compute backend + SIMD tier.
    #[wasm_bindgen(method, js_name = "backend_info")]
    fn backend_info(this: &BarqVWebJs) -> String;

    /// Persist HNSW index to OPFS.
    #[wasm_bindgen(method, js_name = "save")]
    fn save(this: &BarqVWebJs) -> Promise;

    /// Restore HNSW index from OPFS.
    #[wasm_bindgen(method, js_name = "load")]
    fn load(this: &BarqVWebJs) -> Promise;

    /// Wipe the in-memory collection.
    #[wasm_bindgen(method, js_name = "clear")]
    fn clear(this: &BarqVWebJs) -> Promise;
}

// ── BarqMeshWeb — the public WASM class exposed to JavaScript ─────────────

/// `BarqMeshWeb` is the main entry point for Phase 1.
///
/// It combines:
/// - **barq-vweb** (HNSW vector DB, browser-native)
/// - **barq-wasm** (SIMD compute kernels — `vector_normalize`, `mean`, `std_dev`)
///
/// Data flow:
/// ```
/// raw Vec<f32>
///   → barq-wasm::vector_normalize()   (L2 normalise)
///   → pack into js_sys::Float32Array
///   → barq-vweb::insert_vectors()     (HNSW index)
///
/// query Vec<f32>
///   → barq-wasm::vector_normalize()
///   → barq-vweb::search_vector()      (kNN)
///   → Vec<SearchResult { id, score }>
/// ```
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BarqMeshWeb {
    store: BarqVWebJs,
    collection: String,
    dim: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl BarqMeshWeb {
    // ── Constructor ─────────────────────────────────────────────────────────

    /// Create a new mesh instance backed by `barq-vweb`.
    ///
    /// - `collection_name`: name of the HNSW collection.
    /// - `dim`: embedding dimension (must match inserted vectors).
    #[wasm_bindgen(constructor)]
    pub fn new(collection_name: &str, dim: usize) -> BarqMeshWeb {
        console_error_panic_hook::set_once();
        let store = BarqVWebJs::new(collection_name, JsValue::NULL);
        BarqMeshWeb {
            store,
            collection: collection_name.to_string(),
            dim,
        }
    }

    // ── Insert ───────────────────────────────────────────────────────────────

    /// Normalise `raw_vec` with barq-wasm SIMD, then insert into barq-vweb HNSW.
    ///
    /// Returns the new total vector count.
    pub async fn upsert_vector(
        &self,
        raw_vec: Vec<f32>,
        id: u32,
    ) -> Result<usize, JsValue> {
        // ① barq-wasm: L2 normalise (ensures unit vectors for cosine search)
        let normalised = vector_normalize(&raw_vec);

        // ② Pack into JS typed arrays
        let flat = Float32Array::from(normalised.as_slice());
        let ids = Uint32Array::from(&[id][..]);

        // ③ barq-vweb: insert into HNSW
        JsFuture::from(self.store.insert_vectors(&flat, &ids, self.dim)).await?;

        Ok(self.store.count())
    }

    /// Batch insert: normalise each vector, then call barq-vweb once.
    ///
    /// - `flat_vecs`: all vectors packed flat (length = n × dim)
    /// - `ids`      : u32 ID for each vector (length = n)
    ///
    /// Returns the new total vector count.
    pub async fn upsert_vectors(
        &self,
        flat_vecs: Vec<f32>,
        ids: Vec<u32>,
    ) -> Result<usize, JsValue> {
        let n = ids.len();
        let dim = self.dim;

        // ① barq-wasm: normalise each vector in-place
        let mut normalised_flat = Vec::with_capacity(flat_vecs.len());
        for i in 0..n {
            let start = i * dim;
            let end = start + dim;
            if end <= flat_vecs.len() {
                let norm = vector_normalize(&flat_vecs[start..end]);
                normalised_flat.extend_from_slice(&norm);
            }
        }

        // ② Pack into JS typed arrays
        let flat_js = Float32Array::from(normalised_flat.as_slice());
        let ids_js = Uint32Array::from(ids.as_slice());

        // ③ barq-vweb: single batch insert → HNSW
        JsFuture::from(self.store.insert_vectors(&flat_js, &ids_js, dim)).await?;

        Ok(self.store.count())
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Normalise `query_vec` then run kNN search via barq-vweb.
    ///
    /// Returns a JSON string of `[{ id, score }]`.
    pub async fn search_vector(
        &self,
        query_vec: Vec<f32>,
        top_k: usize,
    ) -> Result<String, JsValue> {
        // ① barq-wasm: normalise query
        let normalised = vector_normalize(&query_vec);
        let query_js = Float32Array::from(normalised.as_slice());

        // ② barq-vweb: kNN search
        let js_result = JsFuture::from(self.store.search_vector(&query_js, top_k)).await?;

        // ③ Return as JSON string (JS caller can JSON.parse)
        let json = js_sys::JSON::stringify(&js_result)
            .map(|s| s.as_string().unwrap_or_default())
            .unwrap_or_default();
        Ok(json)
    }

    /// Insert texts into barq-vweb, automatically computing MiniLM embeddings + BM25 index.
    pub async fn ingest_texts(
        &self,
        texts: Vec<String>,
    ) -> Result<usize, JsValue> {
        let texts_arr = Array::new();
        for t in texts {
            texts_arr.push(&JsValue::from_str(&t));
        }
        let metadata_arr = Array::new(); // placeholder metadata
        
        JsFuture::from(self.store.insert_texts(&texts_arr, &metadata_arr)).await?;
        Ok(self.count())
    }

    /// Hybrid or dense text search via barq-vweb.
    pub async fn retrieve_hybrid(
        &self,
        query: String,
        top_k: usize,
    ) -> Result<String, JsValue> {
        let js_result = JsFuture::from(self.store.search(query, top_k, true)).await?;
        let json = js_sys::JSON::stringify(&js_result)
            .map(|s| s.as_string().unwrap_or_default())
            .unwrap_or_default();
        Ok(json)
    }

    // ── Embedding stats ──────────────────────────────────────────────────────

    /// Compute embedding quality stats for `raw_vec` using barq-wasm.
    ///
    /// Returns `{ mean, std_dev, norm, dim }` as a JSON string.
    pub fn embedding_stats(&self, raw_vec: Vec<f32>) -> String {
        let m = mean(&raw_vec);
        let s = std_dev(&raw_vec);
        let n = vector_norm_simd(&raw_vec);
        let stats = EmbeddingStats {
            mean: m,
            std_dev: s,
            norm: n,
            dim: raw_vec.len(),
        };
        serde_json::to_string(&stats).unwrap_or_default()
    }

    // ── Utility ──────────────────────────────────────────────────────────────

    /// Number of vectors currently indexed.
    pub fn count(&self) -> usize {
        self.store.count()
    }

    /// Active compute backend + SIMD tier string from barq-vweb.
    pub fn backend_info(&self) -> String {
        self.store.backend_info()
    }

    /// Persist the HNSW index to OPFS.
    pub async fn save(&self) -> Result<String, JsValue> {
        let result = JsFuture::from(self.store.save()).await?;
        Ok(result.as_string().unwrap_or_else(|| "saved".to_string()))
    }

    /// Restore the HNSW index from OPFS.
    pub async fn load(&self) -> Result<String, JsValue> {
        let result = JsFuture::from(self.store.load()).await?;
        Ok(result.as_string().unwrap_or_else(|| "loaded".to_string()))
    }

    /// Wipe the in-memory collection.
    pub async fn clear(&self) -> Result<(), JsValue> {
        JsFuture::from(self.store.clear()).await?;
        Ok(())
    }

    /// Name of the current collection.
    pub fn collection_name(&self) -> String {
        self.collection.clone()
    }

    /// Configured embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
