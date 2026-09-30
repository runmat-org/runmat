use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{Geometry, PoolError};

pub(in crate::builtins::deep_learning) struct PoolOutput {
    pub tensor: Tensor,
    pub axes: AxisPlan,
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    axes: &AxisPlan,
) -> Result<PoolOutput, PoolError> {
    let geometry = Geometry::new(axes)?;
    geometry.validate_input(input)?;
    let tensor = if let Some(values) = input.as_f32_slice() {
        Tensor::from_numeric_storage(
            NumericStorage::F32(forward_values(values, &geometry)),
            geometry.output.shape().to_vec(),
        )
    } else if let Some(values) = input.as_f64_slice() {
        Tensor::from_numeric_storage(
            NumericStorage::F64(forward_values(values, &geometry)),
            geometry.output.shape().to_vec(),
        )
    } else {
        return Err(PoolError::UnsupportedPrecision);
    }
    .map_err(PoolError::Storage)?;
    Ok(PoolOutput {
        tensor,
        axes: geometry.output,
    })
}

fn forward_values<T: Real>(values: &[T], geometry: &Geometry) -> Vec<T> {
    let mut output = vec![T::zero(); geometry.output.element_count()];
    let divisor = T::from_f64(geometry.pool_size as f64);
    for (output_linear, target) in output.iter_mut().enumerate() {
        let base = geometry.input_base_offset(output_linear);
        let mut sum = T::zero();
        let mut finite_overflow = false;
        for pool_index in 0..geometry.pool_size {
            let value = values[geometry.input_offset(base, pool_index)];
            let next = sum + value;
            if sum.is_finite() && value.is_finite() && !next.is_finite() {
                finite_overflow = true;
                break;
            }
            sum = next;
        }
        if finite_overflow {
            // Recompute only the exceptional finite range case. The ordinary
            // sum/divide path keeps the existing rounding and throughput.
            let mut mean = T::zero();
            for pool_index in 0..geometry.pool_size {
                let count = T::from_f64((pool_index + 1) as f64);
                let previous_weight = T::from_f64(pool_index as f64) / count;
                mean = mean * previous_weight
                    + values[geometry.input_offset(base, pool_index)] / count;
            }
            *target = mean;
        } else {
            *target = sum / divisor;
        }
    }
    output
}
