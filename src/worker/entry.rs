use wasm_bindgen::prelude::*;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, MessagePort};
use std::rc::Rc;
use std::cell::RefCell;

use crate::mesh::types::{AiMeshTask, AiMeshResult, TaskType, TaskPayload, ComputeOp};
use barq_wasm::wasm_bindings::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn worker_entry_point(worker_id: usize, port: MessagePort) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let scope: DedicatedWorkerGlobalScope = js_sys::global().dyn_into().unwrap();
    
    let port_rc = Rc::new(port);
    let port_rc_clone = port_rc.clone();

    let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
        if let Some(msg_str) = event.data().as_string() {
            if let Ok(task) = serde_json::from_str::<AiMeshTask>(&msg_str) {
                let start_time = js_sys::Date::now();

                let result = run_task(&task);

                let end_time = js_sys::Date::now();

                let mut res = result;
                res.latency_ms = end_time - start_time;

                let res_json = serde_json::to_string(&res).unwrap();
                let _ = port_rc_clone.post_message(&JsValue::from_str(&res_json));
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);

    // Keep closure alive
    let onmsg_ref = onmessage.as_ref().unchecked_ref();
    // We actually need to assign it to port.onmessage
    port_rc.clone().set_onmessage(Some(onmsg_ref));
    
    // Leak the closure so it stays alive for the worker lifespan
    onmessage.forget();
    
    Ok(())
}

fn run_task(task: &AiMeshTask) -> AiMeshResult {
    let mut success = true;
    let mut output = serde_json::Value::Null;
    let mut error = None;

    match &task.payload {
        // Phase 1 - basic (no worker needed for these but included for completeness)
        TaskPayload::RawVector { data } => {
            let norm = vector_normalize(data);
            output = serde_json::json!({ "normalised": norm });
        }
        TaskPayload::RawBytes { data } => {
            output = serde_json::json!({ "bytes_len": data.len() });
        }

        // Phase 2 features
        TaskPayload::Ingest { embedding } => {
            // Worker doesn't call barq-vweb insert directly here (JS object ownership issues across threads),
            // Instead we normalise and send back for the main thread to do the barq-vweb insert!
            let norm = vector_normalize(embedding);
            output = serde_json::json!({ "normalised": norm });
        }
        TaskPayload::Embed { inputs } => {
            let norm = vector_normalize(inputs);
            let m = mean(inputs);
            let s = std_dev(inputs);
            let raw_norm = vector_norm_simd(inputs);
            output = serde_json::json!({
                "normalised": norm,
                "mean": m,
                "std_dev": s,
                "norm": raw_norm
            });
        }
        TaskPayload::Compute { op, a, b } => {
            match op {
                ComputeOp::DotProduct => {
                    let dot = dot_product_simd(a, b);
                    output = serde_json::json!({ "result": dot });
                }
                ComputeOp::MatrixMultiply => {
                    // Assuming square matrix of size sqrt(len)
                    let n = (a.len() as f64).sqrt() as usize;
                    let result_mat = matrix_multiply_tiled(a, b, n);
                    output = serde_json::json!({ "result": result_mat });
                }
                ComputeOp::Relu => {
                    let res = relu(a);
                    output = serde_json::json!({ "result": res });
                }
                ComputeOp::Softmax => {
                    let res = softmax(a);
                    output = serde_json::json!({ "result": res });
                }
                ComputeOp::Sigmoid => {
                    let res = sigmoid(a);
                    output = serde_json::json!({ "result": res });
                }
            }
        }
        TaskPayload::Compress { bytes } => {
            let compressed = lz4_compress_optimized(bytes);
            output = serde_json::json!({ "compressed_bytes": compressed });
        }
        TaskPayload::DedupCheck { a, b } => {
            let norm_a = vector_normalize(a);
            let norm_b = vector_normalize(b);
            let sim = cosine_similarity_simd(&norm_a, &norm_b);
            output = serde_json::json!({ "similarity": sim, "is_duplicate": sim > 0.98 });
        }
    }

    AiMeshResult {
        task_id: task.id.clone(),
        success,
        output,
        error,
        latency_ms: 0.0,
    }
}
