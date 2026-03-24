use wasm_bindgen::prelude::*;
use web_sys::{BroadcastChannel, MessageEvent};
use std::rc::Rc;
use std::cell::RefCell;
use crate::mesh::types::AiMeshResult;

pub struct MeshBus {
    channel: BroadcastChannel,
    _closures: Vec<Closure<dyn FnMut(MessageEvent)>>,
}

impl MeshBus {
    /// Create un-configured bus
    pub fn new(channel_name: &str) -> Self {
        let channel = BroadcastChannel::new(channel_name).unwrap();
        
        Self {
            channel,
            _closures: Vec::new(),
        }
    }

    /// Publish task result to all tabs
    pub fn publish(&self, result: &AiMeshResult) -> Result<(), JsValue> {
        if let Ok(json) = serde_json::to_string(result) {
            self.channel.post_message(&JsValue::from_str(&json))?;
        }
        Ok(())
    }

    /// Subscribe to results coming from any tab
    pub fn on_result<F>(&mut self, mut callback: F)
    where
        F: FnMut(AiMeshResult) + 'static,
    {
        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Some(str_val) = event.data().as_string() {
                if let Ok(res) = serde_json::from_str::<AiMeshResult>(&str_val) {
                    callback(res);
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        self.channel.set_onmessage(Some(closure.as_ref().unchecked_ref()));
        self._closures.push(closure); // Keep alive
    }
}
