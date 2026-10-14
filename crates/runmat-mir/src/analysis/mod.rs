mod dataflow;
mod constructor_entry;
mod engine;
mod fusion_stage;
mod inference;
mod parallel;
mod reachability;
mod regions;
mod spawn_safety;
mod store;

pub use engine::analyze_assembly;
pub(crate) use engine::lexical_reference_graph;
pub use fusion_stage::{FusionStageContract, FusionStageFactStatus};
pub use reachability::*;
pub use store::*;
