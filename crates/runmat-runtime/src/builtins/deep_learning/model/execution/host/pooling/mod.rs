//! Global 1-D average pooling and its exact axis adjoint.

mod backward;
mod complex;
mod geometry;
mod kernel;
mod mean;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use complex::{backward_complex, forward_complex};
pub(in crate::builtins::deep_learning) use geometry::PoolError;
pub(in crate::builtins::deep_learning) use kernel::forward;

pub(in crate::builtins::deep_learning) fn admit(
    axes: &crate::builtins::deep_learning::model::plan::axes::AxisPlan,
) -> Result<crate::builtins::deep_learning::model::plan::axes::AxisPlan, PoolError> {
    Ok(geometry::Geometry::new(axes)?.output)
}

#[cfg(test)]
mod complex_tests;
#[cfg(test)]
mod tests;
