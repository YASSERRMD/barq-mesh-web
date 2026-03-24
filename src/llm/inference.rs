use wasm_bindgen::prelude::*;
use crate::mesh::aimesh::AiMesh;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum LlmProvider {
    WebLlm,     // Offline via @mlc-ai/web-llm
    OpenRouter, // Online fallback
}

#[cfg(feature = "wasm")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LlmRequest {
    pub provider: LlmProvider,
    pub prompt: String,
    pub temperature: f32,
    pub use_rag: bool,
    pub collection_context: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct LlmRouter {
    provider: LlmProvider,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl LlmRouter {
    #[wasm_bindgen(constructor)]
    pub fn new(provider_str: &str) -> Self {
        let provider = match provider_str {
            "webllm" => LlmProvider::WebLlm,
            _ => LlmProvider::OpenRouter,
        };
        LlmRouter { provider }
    }

    /// Prepares an LLM request by augmenting the prompt with RAG context if enabled.
    /// The actual model execution happens in JavaScript due to WebGPU / Fetch API ease.
    #[wasm_bindgen]
    pub async fn prepare_rag_prompt(
        &self,
        mesh: &AiMesh,
        prompt: String,
    ) -> Result<String, JsValue> {
        let mut final_prompt = prompt.clone();

        // Use mesh hybrid search for RAG
        if let Ok(results_json) = mesh.retrieve_hybrid(prompt.clone(), 3).await {
            // Simplified: The actual integration would parse JSON and construct exactly the injected context
            // Here we assume the frontend can extract `{ id, score }` and append the text.
            final_prompt = format!(
                "Context from Vector DB (top 3 matches):\n{}\n\nUser Question:\n{}",
                results_json, prompt
            );
        }

        let req = LlmRequest {
            provider: self.provider.clone(),
            prompt: final_prompt,
            temperature: 0.7,
            use_rag: true,
            collection_context: "RAG Augmented".to_string(),
        };

        Ok(serde_json::to_string(&req).unwrap_or_default())
    }

    /// Semantic Verification of LLM Output via barq-wasm
    #[wasm_bindgen]
    pub fn verify_output_semantically(
        &self,
        expected_embedding_json: String,
        actual_embedding_json: String,
    ) -> f64 {
        // Parse the float arrays and run SIMD cosine similarity
        let expected: Vec<f32> = serde_json::from_str(&expected_embedding_json).unwrap_or_default();
        let actual: Vec<f32> = serde_json::from_str(&actual_embedding_json).unwrap_or_default();
        
        if expected.is_empty() || actual.is_empty() { return 0.0; }

        let score = barq_wasm::wasm_bindings::cosine_similarity_simd(&expected, &actual);
        score as f64
    }
}
