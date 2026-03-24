//! barq-mesh-web — browser-native distributed AI agent mesh
//!
//! Phase 1: Core Foundation
//! - Wires barq-wasm (SIMD compute) + barq-vweb (browser vector DB)
//! - Exports `BarqMeshWeb` to JavaScript
//! - Proves both crates compile to a single WASM binary

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

pub mod mesh;
pub mod store;
pub mod worker;
pub mod agents;
pub mod llm;
pub mod mcp;

/// Called automatically when the WASM module is instantiated.
/// Sets up the panic hook so Rust panics surface in the browser console.
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

// ── Top-level re-exports so JS sees BarqMeshWeb directly ──────────────────
#[cfg(feature = "wasm")]
pub use store::barq_vweb_store::BarqMeshWeb;
#[cfg(feature = "wasm")]
pub use mesh::aimesh::AiMesh;
#[cfg(feature = "wasm")]
pub use llm::inference::LlmRouter;
#[cfg(feature = "wasm")]
pub use mesh::topology::TopologyManager;
#[cfg(feature = "wasm")]
pub use mcp::server::McpServer;
