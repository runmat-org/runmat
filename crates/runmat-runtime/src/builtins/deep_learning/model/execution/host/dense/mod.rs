//! Axis-aware fully connected projection and its native-precision adjoint.

mod geometry;
mod kernel;
mod backward;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use geometry::admit;
pub(in crate::builtins::deep_learning) use geometry::DenseError;
pub(in crate::builtins::deep_learning) use kernel::forward;

#[cfg(test)]
mod tests;
