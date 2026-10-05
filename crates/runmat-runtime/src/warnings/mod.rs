//! Session-scoped warning policy and emission.

mod emission;
mod policy;

#[cfg(test)]
mod tests;

pub(crate) use emission::{emit, WarningRequest};
pub(crate) use policy::{with_policy, WarningPolicy, WarningState};
pub(crate) use runmat_builtins::warning_semantics::WarningMode;
