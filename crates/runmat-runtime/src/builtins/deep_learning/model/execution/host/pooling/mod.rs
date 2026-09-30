//! Global 1-D average pooling and its exact axis adjoint.

mod geometry;
mod kernel;
mod backward;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use geometry::PoolError;
pub(in crate::builtins::deep_learning) use kernel::{forward, PoolOutput};

#[cfg(test)]
mod tests;
