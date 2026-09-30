//! Layer normalization with native-precision statistics and adjoints.

mod kernel;
mod backward;

pub(in crate::builtins::deep_learning) use backward::backward;
pub(in crate::builtins::deep_learning) use crate::builtins::deep_learning::model::plan::normalization::{
    NormalizationError, NormalizationOptions,
};
pub(in crate::builtins::deep_learning) use kernel::{forward, NormalizationStats};

pub(in crate::builtins::deep_learning) fn admit(
    axes: &crate::builtins::deep_learning::model::plan::axes::AxisPlan,
    scale_shape: &[usize],
    offset_shape: &[usize],
    dtype: runmat_value::NumericDType,
    options: NormalizationOptions,
) -> Result<crate::builtins::deep_learning::model::plan::axes::AxisPlan, NormalizationError> {
    let geometry =
        crate::builtins::deep_learning::model::plan::normalization::Geometry::new(axes, options)?;
    geometry.validate_shapes(scale_shape, offset_shape, dtype)?;
    Ok(axes.clone())
}

#[cfg(test)]
mod tests;
