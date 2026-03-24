use wasm_bindgen::prelude::*;
use web_sys::{Worker, WorkerOptions, WorkerType, MessageChannel, MessagePort, MessageEvent};
use std::rc::Rc;
use std::cell::RefCell;

use crate::store::barq_vweb_store::BarqMeshWeb;
use crate::mesh::types::{AiMeshTask, AiMeshResult, TaskStatus};
use crate::mesh::queue::MeshQueue;
use crate::mesh::bus::MeshBus;

// ── Exported Worker Stats for JS
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone, serde::Serialize)]
pub struct WorkerStats {
    pub workers: usize,
    pub busy: usize,
    pub queued: usize,
    pub completed: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WorkerPool {
    #[wasm_bindgen(skip)]
    pub pool: Rc<RefCell<Vec<ChannelBundle>>>,
    #[wasm_bindgen(skip)]
    pub queue: Rc<RefCell<MeshQueue>>,
    #[wasm_bindgen(skip)]
    pub bus: Rc<RefCell<MeshBus>>,
    #[wasm_bindgen(skip)]
    pub completed_count: Rc<RefCell<usize>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WorkerPool {
    #[wasm_bindgen(constructor)]
    pub fn new_js(count: usize) -> Result<WorkerPool, JsValue> {
        let bus = Rc::new(RefCell::new(MeshBus::new("mesh-bus")));
        Self::new(count, bus)
    }

    #[wasm_bindgen]
    pub async fn dispatch_js(&self, task_json: String, store: &BarqMeshWeb) -> Result<(), JsValue> {
        if let Ok(task) = serde_json::from_str::<AiMeshTask>(&task_json) {
            self.dispatch(task, store).await?;
        }
        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        serde_json::to_string(&self.stats()).unwrap()
    }
}

pub struct ChannelBundle {
    pub worker: Worker,
    pub port1: MessagePort,
    pub busy: bool,
    pub closure: Closure<dyn FnMut(MessageEvent)>,
}

impl WorkerPool {
    pub fn new(count: usize, bus: Rc<RefCell<MeshBus>>) -> Result<Self, JsValue> {
        let pool = Rc::new(RefCell::new(Vec::<ChannelBundle>::with_capacity(count)));
        let completed_count = Rc::new(RefCell::new(0));
        let queue = Rc::new(RefCell::new(MeshQueue::new()));

        for i in 0..count {
            let mut opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);

            let worker = Worker::new_with_options("worker.js", &opts)?;
            let channel = MessageChannel::new()?;
            let port1 = channel.port1();
            let port2 = channel.port2();

            // Send initialization data with port2
            let init_msg = js_sys::Object::new();
            js_sys::Reflect::set(&init_msg, &"workerId".into(), &JsValue::from_f64(i as f64))?;
            js_sys::Reflect::set(&init_msg, &"port".into(), &port2)?;
            
            // Post init message pushing the port to the worker.
            let transfer_arr = js_sys::Array::new();
            transfer_arr.push(&port2);
            worker.post_message_with_transfer(&init_msg, &transfer_arr)?;

            let pool_ref = pool.clone();
            let bus_ref = bus.clone();
            let count_ref = completed_count.clone();

            // Setup listener on port1 to receive AiMeshResult
            let onmsg = Closure::wrap(Box::new(move |event: MessageEvent| {
                if let Some(str_val) = event.data().as_string() {
                    if let Ok(result) = serde_json::from_str::<AiMeshResult>(&str_val) {
                        *count_ref.borrow_mut() += 1;
                        
                        // Mark worker as free
                        if let Some(b) = pool_ref.borrow_mut().get_mut(i) {
                            b.busy = false;
                        }
                        
                        // Publish result to bus
                        let _ = bus_ref.borrow().publish(&result);
                    }
                }
            }) as Box<dyn FnMut(MessageEvent)>);

            port1.set_onmessage(Some(onmsg.as_ref().unchecked_ref()));

            pool.borrow_mut().push(ChannelBundle {
                worker,
                port1,
                busy: false,
                closure: onmsg,
            });
        }

        Ok(Self {
            pool,
            queue,
            bus,
            completed_count,
        })
    }

    /// Enqueue a task (dedup checks require the store)
    pub async fn dispatch(&self, task: AiMeshTask, store: &BarqMeshWeb) -> Result<(), JsValue> {
        let mut q = self.queue.borrow_mut();
        if q.enqueue(task, store).await? {
            // Task actually added (not deduped), drop queue ref
            drop(q);
            // Trigger drain
            self.drain_loop();
        }
        Ok(())
    }

    /// Try to dispatch pending tasks to idle workers
    pub fn drain_loop(&self) {
        let mut q = self.queue.borrow_mut();
        let mut p = self.pool.borrow_mut();

        for bundle in p.iter_mut() {
            if !bundle.busy {
                if let Some(mut task) = q.dequeue() {
                    // Start task
                    task.status = TaskStatus::Running;
                    let ts_str = serde_json::to_string(&task).unwrap();
                    let _ = bundle.port1.post_message(&JsValue::from_str(&ts_str));
                    bundle.busy = true;
                }
            }
        }
    }

    /// Check stats
    pub fn stats(&self) -> WorkerStats {
        let p = self.pool.borrow();
        WorkerStats {
            workers: p.len(),
            busy: p.iter().filter(|b| b.busy).count(),
            queued: self.queue.borrow().len(),
            completed: *self.completed_count.borrow(),
        }
    }
}
