//! Public descriptors of the typed study-contract classes, one module per
//! class family. Catalog leaves and runtime bindings both read these.

pub mod analysis_sequence;
pub mod compression_policy;
pub mod independent_domain;
pub mod monitor_cadence;
pub mod monitor_stop_condition;
pub mod observation;
pub mod observation_cost;
pub mod observation_event;
pub mod observation_storage;
pub mod observation_target;
pub mod procedure_step;
pub mod reduction_policy;
pub mod restart_policy;
pub mod retention_policy;
pub mod sampling_policy;
pub mod sequence_branch;
pub mod sequence_dependency;
pub mod signal_precision;
pub mod state_input;
pub mod state_output;
pub mod state_transition;
