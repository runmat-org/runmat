//! Stable channel-wise softmax for admitted real tensors.

use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{NumericDType, NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::numeric::Real;

#[derive(Debug)]
pub(in crate::builtins::deep_learning) enum SoftmaxError {
    Axes,
    UnsupportedPrecision,
    GradientMismatch,
    Storage(String),
}
impl std::fmt::Display for SoftmaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Axes => f.write_str("softmax requires valid channel geometry"),
            Self::UnsupportedPrecision => {
                f.write_str("softmax requires single or double precision")
            }
            Self::GradientMismatch => f.write_str("softmax gradient differs from forward output"),
            Self::Storage(error) => write!(f, "softmax tensor storage: {error}"),
        }
    }
}
impl std::error::Error for SoftmaxError {}

pub(in crate::builtins::deep_learning) struct ChannelGeometry {
    channels: usize,
    samples: usize,
    channel_stride: usize,
    other_extents: Vec<usize>,
    other_strides: Vec<usize>,
}

impl ChannelGeometry {
    pub fn new(input: &Tensor, axes: &AxisPlan) -> Result<Self, SoftmaxError> {
        if input.shape() != axes.shape() || input.len() != axes.element_count() {
            return Err(SoftmaxError::Axes);
        }
        Self::from_axes(axes, input.numeric_dtype())
    }

    /// Share exact channel geometry between whole-network preflight and the
    /// payload-bearing forward/adjoint kernels.
    pub fn from_axes(axes: &AxisPlan, dtype: NumericDType) -> Result<Self, SoftmaxError> {
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
            return Err(SoftmaxError::UnsupportedPrecision);
        }
        let channel = axes
            .layout()
            .index(DeepLearningAxisLabel::Channel)
            .ok_or(SoftmaxError::Axes)?;
        let channels = axes.shape()[channel];
        if channels == 0 {
            return Err(SoftmaxError::Axes);
        }
        let mut stride = 1usize;
        let mut channel_stride = 0usize;
        let mut other_extents = Vec::new();
        let mut other_strides = Vec::new();
        let mut samples = 1usize;
        for (index, &extent) in axes.shape().iter().enumerate() {
            if index == channel {
                channel_stride = stride;
            } else {
                other_extents.push(extent);
                other_strides.push(stride);
                samples = samples.checked_mul(extent).ok_or(SoftmaxError::Axes)?;
            }
            stride = stride.checked_mul(extent).ok_or(SoftmaxError::Axes)?;
        }
        Ok(Self {
            channels,
            samples,
            channel_stride,
            other_extents,
            other_strides,
        })
    }

    pub(in crate::builtins::deep_learning) fn samples(&self) -> usize {
        self.samples
    }

    pub(in crate::builtins::deep_learning) fn channels(&self) -> usize {
        self.channels
    }

    pub(in crate::builtins::deep_learning) fn base(&self, mut sample: usize) -> usize {
        let mut offset = 0;
        for (&extent, &stride) in self.other_extents.iter().zip(&self.other_strides) {
            offset += (sample % extent) * stride;
            sample /= extent;
        }
        offset
    }
    pub(in crate::builtins::deep_learning) fn offset(&self, base: usize, channel: usize) -> usize {
        base + channel * self.channel_stride
    }
}

pub(in crate::builtins::deep_learning) fn admit(
    axes: &AxisPlan,
    dtype: NumericDType,
) -> Result<(), SoftmaxError> {
    ChannelGeometry::from_axes(axes, dtype).map(|_| ())
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    axes: &AxisPlan,
) -> Result<Tensor, SoftmaxError> {
    let geometry = ChannelGeometry::new(input, axes)?;
    let storage = match input.numeric_dtype() {
        NumericDType::F32 => NumericStorage::F32(forward_values(
            input
                .as_f32_slice()
                .ok_or(SoftmaxError::UnsupportedPrecision)?,
            &geometry,
        )),
        NumericDType::F64 => NumericStorage::F64(forward_values(
            input
                .as_f64_slice()
                .ok_or(SoftmaxError::UnsupportedPrecision)?,
            &geometry,
        )),
        _ => return Err(SoftmaxError::UnsupportedPrecision),
    };
    Tensor::from_numeric_storage(storage, input.shape().to_vec()).map_err(SoftmaxError::Storage)
}

pub(in crate::builtins::deep_learning) fn backward(
    output: &Tensor,
    dy: &Tensor,
    axes: &AxisPlan,
) -> Result<Tensor, SoftmaxError> {
    let geometry = ChannelGeometry::new(output, axes)?;
    if output.shape() != dy.shape() || output.numeric_dtype() != dy.numeric_dtype() {
        return Err(SoftmaxError::GradientMismatch);
    }
    let storage = match output.numeric_dtype() {
        NumericDType::F32 => NumericStorage::F32(backward_values(
            output
                .as_f32_slice()
                .ok_or(SoftmaxError::UnsupportedPrecision)?,
            dy.as_f32_slice()
                .ok_or(SoftmaxError::UnsupportedPrecision)?,
            &geometry,
        )),
        NumericDType::F64 => NumericStorage::F64(backward_values(
            output
                .as_f64_slice()
                .ok_or(SoftmaxError::UnsupportedPrecision)?,
            dy.as_f64_slice()
                .ok_or(SoftmaxError::UnsupportedPrecision)?,
            &geometry,
        )),
        _ => return Err(SoftmaxError::UnsupportedPrecision),
    };
    Tensor::from_numeric_storage(storage, output.shape().to_vec()).map_err(SoftmaxError::Storage)
}

fn forward_values<T: Real>(x: &[T], g: &ChannelGeometry) -> Vec<T> {
    let mut output = vec![T::zero(); x.len()];
    for sample in 0..g.samples {
        let base = g.base(sample);
        let mut maximum = x[g.offset(base, 0)];
        for channel in 1..g.channels {
            let value = x[g.offset(base, channel)];
            if value > maximum {
                maximum = value;
            }
        }
        let mut sum = T::zero();
        for channel in 0..g.channels {
            let offset = g.offset(base, channel);
            let value = (x[offset] - maximum).exp();
            output[offset] = value;
            sum += value;
        }
        for channel in 0..g.channels {
            let offset = g.offset(base, channel);
            output[offset] = output[offset] / sum;
        }
    }
    output
}

fn backward_values<T: Real>(y: &[T], dy: &[T], g: &ChannelGeometry) -> Vec<T> {
    let mut dx = vec![T::zero(); y.len()];
    for sample in 0..g.samples {
        let base = g.base(sample);
        let mut dot = T::zero();
        for channel in 0..g.channels {
            let offset = g.offset(base, channel);
            dot += y[offset] * dy[offset];
        }
        for channel in 0..g.channels {
            let offset = g.offset(base, channel);
            dx[offset] = y[offset] * (dy[offset] - dot);
        }
    }
    dx
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::DeepLearningAxisLayout;

    #[test]
    fn non_square_bc_and_cb_use_semantic_channel_axis() {
        let bc = Tensor::new(vec![1., 4., 2., 5., 3., 6.], vec![2, 3]).unwrap();
        let axes_bc =
            AxisPlan::new(DeepLearningAxisLayout::parse("BC", 2).unwrap(), bc.shape()).unwrap();
        let cb = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]).unwrap();
        let axes_cb =
            AxisPlan::new(DeepLearningAxisLayout::parse("CB", 2).unwrap(), cb.shape()).unwrap();
        let y_bc = forward(&bc, &axes_bc).unwrap();
        let y_cb = forward(&cb, &axes_cb).unwrap();
        let bc_values = y_bc.as_f64_slice().unwrap();
        let cb_values = y_cb.as_f64_slice().unwrap();
        for (lhs, rhs) in [(0, 0), (2, 1), (4, 2), (1, 3), (3, 4), (5, 5)] {
            assert!((bc_values[lhs] - cb_values[rhs]).abs() < 1e-14);
        }
        let dy = Tensor::new(vec![2.; 6], vec![3, 2]).unwrap();
        assert!(backward(&y_cb, &dy, &axes_cb)
            .unwrap()
            .as_f64_slice()
            .unwrap()
            .iter()
            .all(|v| v.abs() < 1e-14));
    }

    #[test]
    fn nonfinite_channel_rows_propagate_nan_at_both_precisions() {
        let shape = [2, 1];
        let axes = AxisPlan::new(DeepLearningAxisLayout::parse("CB", 2).unwrap(), &shape).unwrap();
        for values in [[f64::NAN, 1.0], [f64::INFINITY, 0.0]] {
            let double = Tensor::new(values.to_vec(), shape.to_vec()).unwrap();
            let single =
                Tensor::from_f32(values.map(|value| value as f32).to_vec(), shape.to_vec())
                    .unwrap();
            assert!(forward(&double, &axes)
                .unwrap()
                .as_f64_slice()
                .unwrap()
                .iter()
                .all(|value| value.is_nan()));
            assert!(forward(&single, &axes)
                .unwrap()
                .as_f32_slice()
                .unwrap()
                .iter()
                .all(|value| value.is_nan()));
        }
    }
}
