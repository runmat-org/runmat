mod dataflow;
mod engine;
mod inference;
mod reachability;
mod regions;
mod spawn_safety;
mod store;

pub use engine::analyze_assembly;
pub use reachability::*;
pub use store::*;
