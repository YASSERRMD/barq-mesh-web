use wasm_bindgen::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;
use web_sys::{BroadcastChannel, MessageEvent};

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct TopologyManager {
    _id: String,
    channel: BroadcastChannel,
    is_leader: Rc<RefCell<bool>>,
    _closure: Closure<dyn FnMut(MessageEvent)>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl TopologyManager {
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str) -> Result<TopologyManager, JsValue> {
        let channel = BroadcastChannel::new("mesh-topology")?;
        
        let is_leader = Rc::new(RefCell::new(false));
        let is_leader_clone = Rc::clone(&is_leader);
        
        // Listen for leader heartbeats or claim messages
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Some(msg) = event.data().as_string() {
                if msg.starts_with("LEADER_CLAIM:") {
                    // Another tab claimed leadership, step down
                    *is_leader_clone.borrow_mut() = false;
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        channel.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        Ok(TopologyManager {
            _id: id.to_string(),
            channel,
            is_leader,
            _closure: onmessage,
        })
    }

    /// Claim leadership over OPFS persisting.
    #[wasm_bindgen]
    pub fn claim_leadership(&self) -> Result<(), JsValue> {
        *self.is_leader.borrow_mut() = true;
        let msg = format!("LEADER_CLAIM:{}", self._id);
        self.channel.post_message(&JsValue::from_str(&msg))?;
        Ok(())
    }

    /// Retrieve local leadership status
    #[wasm_bindgen]
    pub fn is_leader(&self) -> bool {
        *self.is_leader.borrow()
    }
}
