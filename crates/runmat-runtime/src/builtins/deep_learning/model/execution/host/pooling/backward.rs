use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{Geometry, PoolError};

pub(in crate::builtins::deep_learning) fn backward(
    output_gradient: &Tensor,
    input_axes: &AxisPlan,
) -> Result<Tensor, PoolError> {
    let geometry = Geometry::new(input_axes)?;
    if output_gradient.shape() != geometry.output.shape()
        || output_gradient.len() != geometry.output.element_count()
    {
        return Err(PoolError::ShapeMismatch);
    }
    let gradient = if let Some(values) = output_gradient.as_f32_slice() {
        NumericStorage::F32(backward_values(values, &geometry))
    } else if let Some(values) = output_gradient.as_f64_slice() {
        NumericStorage::F64(backward_values(values, &geometry))
    } else {
        return Err(PoolError::UnsupportedPrecision);
    };
    Tensor::from_numeric_storage(gradient, geometry.input.shape().to_vec())
        .map_err(PoolError::Storage)
}

fn backward_values<T: Real>(values: &[T], geometry: &Geometry) -> Vec<T> {
    let mut gradient = vec![T::zero(); geometry.input.element_count()];
    let divisor = T::from_f64(geometry.pool_size as f64);
    for (output_linear, value) in values.iter().enumerate() {
        let base = geometry.input_base_offset(output_linear);
        let contribution = *value / divisor;
        for pool_index in 0..geometry.pool_size {
            gradient[geometry.input_offset(base, pool_index)] += contribution;
        }
    }
    gradient
}
