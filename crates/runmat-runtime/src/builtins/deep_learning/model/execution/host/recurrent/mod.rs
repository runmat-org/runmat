//! Native-precision recurrent host kernels. The network owner supplies admitted
//! options and canonical parameters/state; this module never mutates a network.

mod backward;
mod forward;
mod gates;
mod geometry;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use forward::{forward, RecurrentBackwardCache};
pub(in crate::builtins::deep_learning) use gates::RecurrentGateRows;
pub(in crate::builtins::deep_learning) use geometry::{
    RecurrentError, RecurrentOptions, RecurrentParameterShapes,
};

pub(in crate::builtins::deep_learning) fn admit(
    axes: &crate::builtins::deep_learning::model::plan::axes::AxisPlan,
    parameters: RecurrentParameterShapes<'_>,
    dtype: runmat_value::NumericDType,
    options: RecurrentOptions,
) -> Result<crate::builtins::deep_learning::model::plan::axes::AxisPlan, RecurrentError> {
    Ok(geometry::Geometry::from_shapes(axes, parameters, dtype, options)?.output_axes)
}

fn required_slice<T>(slice: Option<Option<&[T]>>) -> Result<Option<&[T]>, RecurrentError> {
    slice
        .map(|value| value.ok_or(RecurrentError::UnsupportedPrecision))
        .transpose()
}

#[cfg(test)]
mod tests;
