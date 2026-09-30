//! Axis-aware fully connected projection and its native-precision adjoint.

mod geometry;
mod kernel;
mod backward;

pub(in crate::builtins::deep_learning) use backward::{backward, DenseGradients};
pub(in crate::builtins::deep_learning) use geometry::DenseError;
pub(in crate::builtins::deep_learning) use kernel::{forward, DenseOutput};

#[cfg(test)]
mod tests;
