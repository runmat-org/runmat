mod dataflow;
mod engine;
mod facts;
mod inference;
mod spawn_safety;
mod store;

pub use engine::analyze_assembly;
pub use facts::*;
pub use store::*;
