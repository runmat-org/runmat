use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{ConvError, ConvOptions, Geometry, PaddingValue};
use super::kernel::Real;

pub(in crate::builtins::deep_learning) struct ConvGradients {
    pub input: Tensor,
    pub weights: Tensor,
    pub bias: Tensor,
}

pub(in crate::builtins::deep_learning) fn backward(
    input: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
    output_gradient: &Tensor,
    axes: &AxisPlan,
    options: ConvOptions,
) -> Result<ConvGradients, ConvError> {
    let geometry = Geometry::new(input, weights, bias, axes, options)?;
    if output_gradient.shape() != geometry.output_shape.as_slice() {
        return Err(ConvError::Invalid(
            "output gradient shape must equal convolution output shape",
        ));
    }
    let (input_grad, weight_grad, bias_grad) = match (
        input.as_f32_slice(),
        weights.as_f32_slice(),
        output_gradient.as_f32_slice(),
    ) {
        (Some(x), Some(w), Some(dy)) => {
            let (dx, dw, db) = backward_values(x, w, dy, &geometry);
            (
                NumericStorage::F32(dx),
                NumericStorage::F32(dw),
                NumericStorage::F32(db),
            )
        }
        _ => match (
            input.as_f64_slice(),
            weights.as_f64_slice(),
            output_gradient.as_f64_slice(),
        ) {
            (Some(x), Some(w), Some(dy)) => {
                let (dx, dw, db) = backward_values(x, w, dy, &geometry);
                (
                    NumericStorage::F64(dx),
                    NumericStorage::F64(dw),
                    NumericStorage::F64(db),
                )
            }
            _ => return Err(ConvError::UnsupportedPrecision),
        },
    };
    Ok(ConvGradients {
        input: Tensor::from_numeric_storage(input_grad, input.shape().to_vec())
            .map_err(ConvError::Storage)?,
        weights: Tensor::from_numeric_storage(weight_grad, weights.shape().to_vec())
            .map_err(ConvError::Storage)?,
        bias: Tensor::from_numeric_storage(bias_grad, bias.shape().to_vec())
            .map_err(ConvError::Storage)?,
    })
}

fn backward_values<T: Real>(x: &[T], w: &[T], dy: &[T], g: &Geometry) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut dx = vec![T::zero(); x.len()];
    let mut dw = vec![T::zero(); w.len()];
    let mut db = vec![T::zero(); g.output_channels];
    let pad = match g.options.padding_value {
        PaddingValue::Constant(value) => T::from_f64(value),
        _ => T::zero(),
    };
    for time in 0..g.time_size {
        for batch in 0..g.batch_size {
            for out_channel in 0..g.output_channels {
                for out_span in 0..g.output_len {
                    let gradient = dy[g.output_offset(out_span, out_channel, batch, time)];
                    db[out_channel] += gradient;
                    for in_channel in 0..g.input_channels {
                        for filter in 0..g.filter_size {
                            let weight_index = g.weight_offset(filter, in_channel, out_channel);
                            if let Some(span) = g.mapped_span(out_span, filter) {
                                let input_index = g.input_offset(span, in_channel, batch, time);
                                dw[weight_index] += gradient * x[input_index];
                                dx[input_index] += gradient * w[weight_index];
                            } else {
                                dw[weight_index] += gradient * pad;
                            }
                        }
                    }
                }
            }
        }
    }
    (dx, dw, db)
}
