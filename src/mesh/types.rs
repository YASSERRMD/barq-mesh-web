//! Phase 1 — Core type definitions for the AI mesh.
//!
//! These types are shared across phases; later phases will extend the
//! `TaskPayload` enum and add new variants without breaking existing code.

use serde::{Deserialize, Serialize};

// ── Task priority ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskPriority {
    High,
    Normal,
    Low,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

// ── Task status ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Deduplicated,
}

impl Default for TaskStatus {
    fn default() -> Self {
        TaskStatus::Pending
    }
}

// ── Task type ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Insert raw float vector into HNSW store.
    Ingest,
    /// Embed + normalise + compute stats for a vector.
    Embed,
    /// Run a raw barq-wasm compute operation.
    Compute,
    /// Compress a byte payload with LZ4.
    Compress,
    /// Check if a vector is a near-duplicate of stored content.
    DedupCheck,
}

// ── Compute operations (used in TaskType::Compute) ────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeOp {
    DotProduct,
    MatrixMultiply,
    Relu,
    Softmax,
    Sigmoid,
}

// ── Agent roles ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Planner,
    Executor,
    Verifier,
    Critic,
}

// ── Plan step (used in Phase 4+) ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: u32,
    pub description: String,
    pub expected_output: String,
}

// ── Artifact variants ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Artifact {
    TaskList { steps: Vec<PlanStep> },
    Execution { output: String },
    Proof { score: f32, passed: bool },
}

// ── TaskPayload — extended per phase ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPayload {
    // Phase 1 — basic raw vector ops
    RawVector { data: Vec<f32> },
    RawBytes { data: Vec<u8> },
    // Phase 2 additions (declared early so mesh/queue.rs compiles cleanly)
    Ingest { embedding: Vec<f32> },
    Embed { inputs: Vec<f32> },
    Compute { op: ComputeOp, a: Vec<f32>, b: Vec<f32> },
    Compress { bytes: Vec<u8> },
    DedupCheck { a: Vec<f32>, b: Vec<f32> },
}

// ── Core task struct ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMeshTask {
    pub id: String,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub payload: TaskPayload,
    pub retry_count: u32,
}

// ── Result returned from workers ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMeshResult {
    pub task_id: String,
    pub success: bool,
    pub output: serde_json::Value,
    pub error: Option<String>,
    /// Latency in milliseconds (set by the worker).
    pub latency_ms: f64,
}

// ── Vector search result (mirrors barq-vweb SearchResult) ─────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u32,
    pub score: f32,
}

// ── Embedding statistics ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    pub mean: f32,
    pub std_dev: f32,
    pub norm: f32,
    pub dim: usize,
}
