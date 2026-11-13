use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{Geometry, PoolError};
use super::mean::stable_mean;

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
    for (output_linear, target) in output.iter_mut().enumerate() {
        let base = geometry.input_base_offset(output_linear);
        *target = stable_mean(geometry.pool_size, |pool_index| {
            values[geometry.input_offset(base, pool_index)]
        });
    }
    output
}
