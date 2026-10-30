mod dataflow;
mod constructor_entry;
mod engine;
mod fusion_stage;
mod inference;
mod parallel;
mod reachability;
mod regions;
mod spawn_safety;
pub use spawn_safety::function_spawn_safety;
mod store;
mod workspace_effect;

pub(crate) use engine::lexical_reference_graph;
pub use engine::{analyze_assembly, diagnose_call_contexts, CallContextDiagnostics};
pub use fusion_stage::{FusionStageContract, FusionStageFactStatus};
pub use reachability::*;
pub use store::*;
