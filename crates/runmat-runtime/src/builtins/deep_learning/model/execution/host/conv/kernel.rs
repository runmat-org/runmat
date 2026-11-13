use std::ops::{Add, AddAssign, Mul};

use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{ConvError, ConvOptions, Geometry, PaddingValue};

pub(super) trait Real: Copy + Add<Output = Self> + AddAssign + Mul<Output = Self> {
    fn zero() -> Self;
    fn from_f64(value: f64) -> Self;
}

impl Real for f32 {
    fn zero() -> Self {
        0.0
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl Real for f64 {
    fn zero() -> Self {
        0.0
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}

pub(in crate::builtins::deep_learning) struct ConvOutput {
    pub tensor: Tensor,
    pub axes: AxisPlan,
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
    axes: &AxisPlan,
    options: ConvOptions,
) -> Result<ConvOutput, ConvError> {
    let geometry = Geometry::new(input, weights, bias, axes, options)?;
    let tensor = match (
        input.as_f32_slice(),
        weights.as_f32_slice(),
        bias.as_f32_slice(),
    ) {
        (Some(x), Some(w), Some(b)) => Tensor::from_numeric_storage(
            NumericStorage::F32(forward_values(x, w, b, &geometry)),
            geometry.output_shape.clone(),
        ),
        _ => match (
            input.as_f64_slice(),
            weights.as_f64_slice(),
            bias.as_f64_slice(),
        ) {
            (Some(x), Some(w), Some(b)) => Tensor::from_numeric_storage(
                NumericStorage::F64(forward_values(x, w, b, &geometry)),
                geometry.output_shape.clone(),
            ),
            _ => return Err(ConvError::UnsupportedPrecision),
        },
    }
    .map_err(ConvError::Storage)?;
    let axes = AxisPlan::new(axes.layout().clone(), &geometry.output_shape)?;
    Ok(ConvOutput { tensor, axes })
}

pub(super) fn forward_values<T: Real>(x: &[T], w: &[T], b: &[T], g: &Geometry) -> Vec<T> {
    let mut output = vec![T::zero(); g.output_elements];
    let pad = match g.options.padding_value {
        PaddingValue::Constant(value) => T::from_f64(value),
        _ => T::zero(),
    };
    for time in 0..g.time_size {
        for batch in 0..g.batch_size {
            for out_channel in 0..g.output_channels {
                for out_span in 0..g.output_len {
                    let mut sum = b[out_channel];
                    for in_channel in 0..g.input_channels {
                        for filter in 0..g.filter_size {
                            let sample = g.mapped_span(out_span, filter).map_or(pad, |span| {
                                x[g.input_offset(span, in_channel, batch, time)]
                            });
                            sum += sample * w[g.weight_offset(filter, in_channel, out_channel)];
                        }
                    }
                    output[g.output_offset(out_span, out_channel, batch, time)] = sum;
                }
            }
        }
    }
    output
}
