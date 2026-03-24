use wasm_bindgen::prelude::*;
use crate::store::barq_vweb_store::BarqMeshWeb;
use crate::mesh::pool::{WorkerPool, WorkerStats};
use crate::mesh::types::{AiMeshTask, TaskStatus, TaskPayload};
use std::rc::Rc;
use std::cell::RefCell;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone)]
pub struct AiMesh {
    // We keep store accessible 
    #[wasm_bindgen(skip)]
    pub store: Rc<BarqMeshWeb>,
    #[wasm_bindgen(skip)]
    pub pool: Rc<WorkerPool>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl AiMesh {
    /// Create the full standalone Mesh (pool + store).
    #[wasm_bindgen]
    pub fn create(workers: usize, collection_name: &str, dim: usize) -> Result<AiMesh, JsValue> {
        console_error_panic_hook::set_once();
        let store = Rc::new(BarqMeshWeb::new(collection_name, dim));
        let pool = Rc::new(WorkerPool::new_js(workers)?);
        
        Ok(AiMesh { store, pool })
    }

    /// Ingest full text strings. Embeddings map automatically via barq-vweb (using its ONNX MiniLM layer).
    #[wasm_bindgen]
    pub async fn ingest_texts(&self, texts_json: String) -> Result<usize, JsValue> {
        // Parse JS String array 
        let texts: Vec<String> = serde_json::from_str(&texts_json)
            .map_err(|e| JsValue::from_str(&format!("JSON Parse Error: {}", e)))?;
        
        self.store.ingest_texts(texts).await
    }

    /// Perform a hybrid search combining BM25 keyword matching + HNSW kNN. Results reranked via RRF.
    #[wasm_bindgen]
    pub async fn retrieve_hybrid(&self, query: String, top_k: usize) -> Result<String, JsValue> {
        self.store.retrieve_hybrid(query, top_k).await
    }

    #[wasm_bindgen]
    pub async fn retrieve(&self, query_vec_json: String, top_k: usize) -> Result<String, JsValue> {
        let q: Vec<f32> = serde_json::from_str(&query_vec_json)
             .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.store.search_vector(q, top_k).await
    }

    /// Save the underlying index to OPFS
    #[wasm_bindgen]
    pub async fn persist(&self) -> Result<String, JsValue> {
        self.store.save().await
    }

    /// Load the underlying index from OPFS
    #[wasm_bindgen]
    pub async fn restore(&self) -> Result<String, JsValue> {
        self.store.load().await
    }

    /// Wipe index memory
    #[wasm_bindgen]
    pub async fn clear(&self) -> Result<(), JsValue> {
        self.store.clear().await
    }

    #[wasm_bindgen]
    pub fn vector_count(&self) -> usize {
        self.store.count()
    }

    #[wasm_bindgen]
    pub fn backend(&self) -> String {
        self.store.backend_info()
    }
    
    /// Pass dispatch explicitly directly through to WorkerPool
    #[wasm_bindgen]
    pub async fn dispatch_task(&self, task_json: String) -> Result<(), JsValue> {
        self.pool.dispatch_js(task_json, &self.store).await
    }

    #[wasm_bindgen]
    pub fn pool_stats(&self) -> String {
        self.pool.get_stats()
    }

    /// Run the Antigravity Agent loop (Planner -> Executor -> Verifier -> Critic)
    #[wasm_bindgen]
    pub async fn run_pipeline(&self, prompt: String) -> Result<String, JsValue> {
        let result = crate::agents::pipeline::run_agent_pipeline(prompt).await?;
        Ok(serde_json::to_string(&result).unwrap_or_default())
    }

    #[wasm_bindgen]
    pub async fn shutdown(&self) -> Result<(), JsValue> {
        // Shutdown pool if required
        Ok(())
    }
}
