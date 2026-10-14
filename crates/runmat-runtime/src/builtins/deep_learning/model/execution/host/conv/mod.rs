//! Learned, multichannel 1-D cross-correlation on authoritative host storage.

mod backward;
mod geometry;
mod kernel;
mod resident;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use geometry::{
    ConvError, ConvOptions, Padding, PaddingValue,
};
pub(in crate::builtins::deep_learning) use kernel::forward;
pub(in crate::builtins::deep_learning) use resident::resident_request;

pub(in crate::builtins::deep_learning) fn admit(
    axes: &crate::builtins::deep_learning::model::plan::axes::AxisPlan,
    weights_shape: &[usize],
    bias_shape: &[usize],
    dtype: runmat_value::NumericDType,
    options: ConvOptions,
) -> Result<crate::builtins::deep_learning::model::plan::axes::AxisPlan, ConvError> {
    let geometry =
        geometry::Geometry::from_shapes(axes, weights_shape, bias_shape, dtype, options)?;
    crate::builtins::deep_learning::model::plan::axes::AxisPlan::new(
        axes.layout().clone(),
        &geometry.output_shape,
    )
    .map_err(ConvError::from)
}

#[cfg(test)]
mod tests;
