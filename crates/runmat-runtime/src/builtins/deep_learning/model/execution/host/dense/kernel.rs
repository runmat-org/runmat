use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{DenseError, Geometry};

pub(in crate::builtins::deep_learning) struct DenseOutput {
    pub tensor: Tensor,
    pub axes: AxisPlan,
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
    axes: &AxisPlan,
) -> Result<DenseOutput, DenseError> {
    let geometry = Geometry::new(input, weights, bias, axes)?;
    let storage = if let (Some(x), Some(w), Some(b)) = (
        input.as_f32_slice(),
        weights.as_f32_slice(),
        bias.as_f32_slice(),
    ) {
        NumericStorage::F32(forward_values(x, w, b, &geometry))
    } else if let (Some(x), Some(w), Some(b)) = (
        input.as_f64_slice(),
        weights.as_f64_slice(),
        bias.as_f64_slice(),
    ) {
        NumericStorage::F64(forward_values(x, w, b, &geometry))
    } else {
        return Err(DenseError::UnsupportedPrecision);
    };
    let tensor = Tensor::from_numeric_storage(storage, geometry.output_axes.shape().to_vec())
        .map_err(DenseError::Storage)?;
    Ok(DenseOutput {
        tensor,
        axes: geometry.output_axes,
    })
}

fn forward_values<T: Real>(x: &[T], w: &[T], b: &[T], g: &Geometry) -> Vec<T> {
    let mut output = vec![T::zero(); g.output_axes.element_count()];
    for sample in 0..g.samples {
        let (input_base, output_base) = g.bases(sample);
        for out_channel in 0..g.output_channels {
            let mut sum = b[out_channel];
            for in_channel in 0..g.input_channels {
                sum += x[g.input_offset(input_base, in_channel)]
                    * w[g.weight_offset(out_channel, in_channel)];
            }
            output[g.output_offset(output_base, out_channel)] = sum;
        }
    }
    output
}
