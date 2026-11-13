//! Complex global average pooling over native single/double components.

use runmat_value::{ComplexElement, ComplexTensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{Geometry, PoolError};
use super::mean::stable_mean;

pub(in crate::builtins::deep_learning) struct ComplexPoolOutput {
    pub tensor: ComplexTensor,
    pub axes: AxisPlan,
}

pub(in crate::builtins::deep_learning) fn forward_complex(
    input: &ComplexTensor,
    axes: &AxisPlan,
) -> Result<ComplexPoolOutput, PoolError> {
    let geometry = Geometry::new(axes)?;
    geometry.validate_complex_input(input)?;
    let shape = geometry.output.shape().to_vec();
    let tensor = if let Some(values) = input.as_f32_slice() {
        ComplexTensor::from_f32(forward_values(values, &geometry), shape)
    } else if let Some(values) = input.as_f64_slice() {
        ComplexTensor::new(forward_values(values, &geometry), shape)
    } else {
        return Err(PoolError::UnsupportedPrecision);
    }
    .map_err(PoolError::Storage)?;
    Ok(ComplexPoolOutput {
        tensor,
        axes: geometry.output,
    })
}

fn forward_values<T: Real>(values: &[ComplexElement<T>], geometry: &Geometry) -> Vec<(T, T)> {
    let mut output = vec![(T::zero(), T::zero()); geometry.output.element_count()];
    for (output_linear, target) in output.iter_mut().enumerate() {
        let base = geometry.input_base_offset(output_linear);
        let real = stable_mean(geometry.pool_size, |pool_index| {
            values[geometry.input_offset(base, pool_index)].0
        });
        let imaginary = stable_mean(geometry.pool_size, |pool_index| {
            values[geometry.input_offset(base, pool_index)].1
        });
        *target = (real, imaginary);
    }
    output
}

/// The real-linear adjoint distributes each real and imaginary component
/// across the pooled axis. The surrounding tape still owns its public complex
/// gradient convention; this host primitive does not choose that policy.
pub(in crate::builtins::deep_learning) fn backward_complex(
    input: &ComplexTensor,
    output_gradient: &ComplexTensor,
    input_axes: &AxisPlan,
) -> Result<ComplexTensor, PoolError> {
    let geometry = Geometry::new(input_axes)?;
    geometry.validate_complex_input(input)?;
    if output_gradient.shape() != geometry.output.shape()
        || output_gradient.len() != geometry.output.element_count()
    {
        return Err(PoolError::ShapeMismatch);
    }
    if output_gradient.numeric_dtype() != input.numeric_dtype() {
        return Err(PoolError::UnsupportedPrecision);
    }
    let shape = geometry.input.shape().to_vec();
    if let Some(values) = output_gradient.as_f32_slice() {
        ComplexTensor::from_f32(backward_values(values, &geometry), shape)
    } else if let Some(values) = output_gradient.as_f64_slice() {
        ComplexTensor::new(backward_values(values, &geometry), shape)
    } else {
        return Err(PoolError::UnsupportedPrecision);
    }
    .map_err(PoolError::Storage)
}

fn backward_values<T: Real>(values: &[ComplexElement<T>], geometry: &Geometry) -> Vec<(T, T)> {
    let mut gradient = vec![(T::zero(), T::zero()); geometry.input.element_count()];
    let divisor = T::from_f64(geometry.pool_size as f64);
    for (output_linear, &ComplexElement(real, imaginary)) in values.iter().enumerate() {
        let base = geometry.input_base_offset(output_linear);
        let contribution = (real / divisor, imaginary / divisor);
        for pool_index in 0..geometry.pool_size {
            gradient[geometry.input_offset(base, pool_index)] = contribution;
        }
    }
    gradient
}
