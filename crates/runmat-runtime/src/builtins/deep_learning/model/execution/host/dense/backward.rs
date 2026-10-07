use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{DenseError, Geometry};

pub(in crate::builtins::deep_learning) struct DenseGradients {
    pub input: Tensor,
    pub weights: Tensor,
    pub bias: Tensor,
}

pub(in crate::builtins::deep_learning) fn backward(
    input: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
    axes: &AxisPlan,
    output_gradient: &Tensor,
) -> Result<DenseGradients, DenseError> {
    let geometry = Geometry::new(input, weights, bias, axes)?;
    if output_gradient.shape() != geometry.output_axes.shape()
        || output_gradient.numeric_dtype() != input.numeric_dtype()
    {
        return Err(DenseError::Invalid(
            "fullyConnectedLayer output gradient shape or class differs",
        ));
    }
    let (dx, dw, db) = if let (Some(x), Some(w), Some(dy)) = (
        input.as_f32_slice(),
        weights.as_f32_slice(),
        output_gradient.as_f32_slice(),
    ) {
        let (dx, dw, db) = backward_values(x, w, dy, &geometry);
        (
            NumericStorage::F32(dx),
            NumericStorage::F32(dw),
            NumericStorage::F32(db),
        )
    } else if let (Some(x), Some(w), Some(dy)) = (
        input.as_f64_slice(),
        weights.as_f64_slice(),
        output_gradient.as_f64_slice(),
    ) {
        let (dx, dw, db) = backward_values(x, w, dy, &geometry);
        (
            NumericStorage::F64(dx),
            NumericStorage::F64(dw),
            NumericStorage::F64(db),
        )
    } else {
        return Err(DenseError::UnsupportedPrecision);
    };
    Ok(DenseGradients {
        input: Tensor::from_numeric_storage(dx, input.shape().to_vec())
            .map_err(DenseError::Storage)?,
        weights: Tensor::from_numeric_storage(dw, weights.shape().to_vec())
            .map_err(DenseError::Storage)?,
        bias: Tensor::from_numeric_storage(db, bias.shape().to_vec())
            .map_err(DenseError::Storage)?,
    })
}

fn backward_values<T: Real>(x: &[T], w: &[T], dy: &[T], g: &Geometry) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut dx = vec![T::zero(); x.len()];
    let mut dw = vec![T::zero(); w.len()];
    let mut db = vec![T::zero(); g.output_channels];
    for sample in 0..g.samples {
        let (input_base, output_base) = g.bases(sample);
        for out_channel in 0..g.output_channels {
            let delta = dy[g.output_offset(output_base, out_channel)];
            db[out_channel] += delta;
            for in_channel in 0..g.input_channels {
                let xi = g.input_offset(input_base, in_channel);
                let wi = g.weight_offset(out_channel, in_channel);
                dx[xi] += delta * w[wi];
                dw[wi] += delta * x[xi];
            }
        }
    }
    (dx, dw, db)
}
