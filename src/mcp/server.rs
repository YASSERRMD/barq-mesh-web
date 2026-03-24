use wasm_bindgen::prelude::*;
use crate::mesh::aimesh::AiMesh;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<String>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct McpServer {
    mesh: AiMesh,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl McpServer {
    #[wasm_bindgen(constructor)]
    pub fn new(mesh: AiMesh) -> Self {
        McpServer { mesh }
    }

    /// Handles incoming MCP JSON-RPC payload asynchronously
    #[wasm_bindgen]
    pub async fn handle_request(&self, request_json: String) -> Result<String, JsValue> {
        let req: McpRequest = match serde_json::from_str(&request_json) {
            Ok(r) => r,
            Err(_) => return Ok(self.build_error("Parse error", -32700, None)),
        };

        if req.jsonrpc != "2.0" {
            return Ok(self.build_error("Invalid Request", -32600, req.id));
        }

        let result = match req.method.as_str() {
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(req.params.unwrap_or_default()).await,
            _ => Err(serde_json::json!({
                "code": -32601,
                "message": "Method not found"
            })),
        };

        match result {
            Ok(res_val) => {
                let resp = McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req.id,
                    result: Some(res_val),
                    error: None,
                };
                Ok(serde_json::to_string(&resp).unwrap_or_default())
            }
            Err(err_val) => {
                let resp = McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: req.id,
                    result: None,
                    error: Some(err_val),
                };
                Ok(serde_json::to_string(&resp).unwrap_or_default())
            }
        }
    }

    fn handle_tools_list(&self) -> Result<serde_json::Value, serde_json::Value> {
        Ok(serde_json::json!({
            "tools": [
                {
                    "name": "barq_run_agent",
                    "description": "Run the Antigravity Agent loop autonomously.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string" }
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "barq_hybrid_search",
                    "description": "Run a hybrid HNSW + BM25 search.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "top_k": { "type": "number" }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }))
    }

    async fn handle_tools_call(&self, params: serde_json::Value) -> Result<serde_json::Value, serde_json::Value> {
        let name = params["name"].as_str().unwrap_or_default();
        let args = params["arguments"].as_object().cloned().unwrap_or_default();

        if name == "barq_run_agent" {
            let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or_default();
            // Call the underlying AI mesh pipeline
            match self.mesh.run_pipeline(prompt.to_string()).await {
                Ok(res_str) => {
                    let parsed: serde_json::Value = serde_json::from_str(&res_str).unwrap_or_default();
                    Ok(serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&parsed).unwrap_or_default()
                        }]
                    }))
                },
                Err(_) => Err(serde_json::json!({"code": -32000, "message": "Pipeline error"}))
            }
        } 
        else if name == "barq_hybrid_search" {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or_default();
            let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            
            match self.mesh.retrieve_hybrid(query.to_string(), top_k).await {
                Ok(res_str) => {
                    Ok(serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": res_str
                        }]
                    }))
                },
                Err(_) => Err(serde_json::json!({"code": -32000, "message": "Search error"}))
            }
        } 
        else {
            Err(serde_json::json!({"code": -32601, "message": "Tool not found"}))
        }
    }

    fn build_error(&self, msg: &str, code: i32, id: Option<String>) -> String {
        let err = McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(serde_json::json!({
                "code": code,
                "message": msg
            })),
        };
        serde_json::to_string(&err).unwrap_or_default()
    }
}
