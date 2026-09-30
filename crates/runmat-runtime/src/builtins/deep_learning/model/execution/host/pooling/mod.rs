//! Global 1-D average pooling and its exact axis adjoint.

mod backward;
mod complex;
mod geometry;
mod kernel;
mod mean;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use complex::{
    backward_complex, forward_complex, ComplexPoolOutput,
};
pub(in crate::builtins::deep_learning) use geometry::PoolError;
pub(in crate::builtins::deep_learning) use kernel::{forward, PoolOutput};

#[cfg(test)]
mod complex_tests;
#[cfg(test)]
mod tests;
