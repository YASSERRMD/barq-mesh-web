pub mod entry;

#[cfg(feature = "wasm")]
pub use entry::worker_entry_point;
