//! Sequential host execution over admitted typed layers.
//!
//! The caller owns whole-network control/geometry/parameter admission before
//! invoking this pure CPU executor. This module has no provider callbacks,
//! does not mutate a network, and does not parse layer names or options.

pub(in crate::builtins::deep_learning) mod backward;
mod forward;
mod types;

pub(in crate::builtins::deep_learning) use backward::{backward, HostBackwardResult};
pub(in crate::builtins::deep_learning) use forward::execute;
pub(in crate::builtins::deep_learning) use types::{
    HostExecutionResult, HostStep, HostTrace, HostValue,
};

#[cfg(test)]
mod tests;
