//! A resident stable normalization stage. Descriptor and whole-output-route
//! admission precede the first parameter upload or provider dispatch.

mod admission;
mod adjoint;
mod execution;
mod validation;
#[cfg(test)]
mod tests;

pub(in crate::builtins::deep_learning) use adjoint::adjoint;
pub(in crate::builtins::deep_learning) use admission::admit_parameters;
pub(in crate::builtins::deep_learning) use execution::forward;
pub(in crate::builtins::deep_learning) use execution::{forward_recorded, NormalizationRecord};
