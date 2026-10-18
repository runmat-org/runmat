//! Source-level stochastic evolution: semantic preparation, whole-call
//! placement, ordered RNG and source publication share one execution path.
mod coordinator;
mod execution;
mod host;
mod host_readback;
mod placement;
mod planning;
mod provider;
mod resources;
mod source_eligibility;

pub use coordinator::{execute_stochastic_source, StochasticSourcePublication};
pub use source_eligibility::{
    positive_stochastic_source_steps, prepare_stochastic_source, StochasticCheckpoint,
    StochasticPreparedSource, StochasticSourceControls, StochasticSourceExecution,
    StochasticTypedPattern, ValueControlRef,
};

#[cfg(test)]
mod tests;
