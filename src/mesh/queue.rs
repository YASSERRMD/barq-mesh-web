use std::collections::VecDeque;
use wasm_bindgen::JsValue;
use serde_json::Value;

use crate::mesh::types::{AiMeshTask, TaskPriority, TaskType, TaskPayload};
use crate::store::barq_vweb_store::BarqMeshWeb;

pub struct MeshQueue {
    high: VecDeque<AiMeshTask>,
    normal: VecDeque<AiMeshTask>,
    low: VecDeque<AiMeshTask>,
}

impl MeshQueue {
    pub fn new() -> Self {
        Self {
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
        }
    }

    /// Enqueue a task, performing semantic deduplication for Ingest tasks.
    pub async fn enqueue(&mut self, mut task: AiMeshTask, store: &BarqMeshWeb) -> Result<bool, JsValue> {
        // Dedup check for ingest
        if task.task_type == TaskType::Ingest {
            if let TaskPayload::Ingest { ref embedding } = task.payload {
                // search_vector parses JSON string
                let result_str = store.search_vector(embedding.clone(), 1).await?;
                if let Ok(results) = serde_json::from_str::<Vec<Value>>(&result_str) {
                    if let Some(first) = results.first() {
                        if let Some(score) = first.get("score").and_then(|s| s.as_f64()) {
                            // High cosine similarity means it's a near duplicate
                            if score > 0.98 {
                                // Task is dropped / rejected as duplicate
                                task.status = crate::mesh::types::TaskStatus::Deduplicated;
                                return Ok(false);
                            }
                        }
                    }
                }
            }
        }

        match task.priority {
            TaskPriority::High => self.high.push_back(task),
            TaskPriority::Normal => self.normal.push_back(task),
            TaskPriority::Low => self.low.push_back(task),
        }
        Ok(true)
    }

    /// Drain queue: High -> Normal -> Low
    pub fn dequeue(&mut self) -> Option<AiMeshTask> {
        self.high.pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }

    pub fn len(&self) -> usize {
        self.high.len() + self.normal.len() + self.low.len()
    }
}
