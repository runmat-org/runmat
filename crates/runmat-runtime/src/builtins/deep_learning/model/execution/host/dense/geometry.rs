use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{NumericDType, Tensor};

use crate::builtins::deep_learning::model::plan::axes::{AxisGeometryError, AxisPlan};

#[derive(Debug)]
pub(in crate::builtins::deep_learning) enum DenseError {
    Axes(AxisGeometryError),
    Invalid(&'static str),
    UnsupportedPrecision,
    Overflow,
    Storage(String),
}

impl std::fmt::Display for DenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Axes(error) => write!(f, "fullyConnectedLayer axes: {error:?}"),
            Self::Invalid(message) => f.write_str(message),
            Self::UnsupportedPrecision => f.write_str("fullyConnectedLayer requires matching real single or double input, Weights, Bias, and gradient"),
            Self::Overflow => f.write_str("fullyConnectedLayer geometry exceeds addressable size"),
            Self::Storage(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for DenseError {}

impl From<AxisGeometryError> for DenseError {
    fn from(error: AxisGeometryError) -> Self {
        Self::Axes(error)
    }
}

pub(super) struct Geometry {
    pub output_axes: AxisPlan,
    pub input_channels: usize,
    pub output_channels: usize,
    pub samples: usize,
    input_channel_stride: usize,
    output_channel_stride: usize,
    sample_shape: Vec<usize>,
    input_sample_strides: Vec<usize>,
    output_sample_strides: Vec<usize>,
}

fn strides(shape: &[usize]) -> Result<Vec<usize>, DenseError> {
    let mut next = 1usize;
    let mut result = Vec::with_capacity(shape.len());
    for &extent in shape {
        result.push(next);
        next = next.checked_mul(extent).ok_or(DenseError::Overflow)?;
    }
    Ok(result)
}

impl Geometry {
    pub fn new(
        input: &Tensor,
        weights: &Tensor,
        bias: &Tensor,
        axes: &AxisPlan,
    ) -> Result<Self, DenseError> {
        if input.shape() != axes.shape() || input.len() != axes.element_count() {
            return Err(DenseError::Invalid(
                "fullyConnectedLayer input and axes disagree",
            ));
        }
        let dtype = input.numeric_dtype();
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64)
            || weights.numeric_dtype() != dtype
            || bias.numeric_dtype() != dtype
        {
            return Err(DenseError::UnsupportedPrecision);
        }
        Self::from_shapes(axes, weights.shape(), bias.shape())
    }

    /// The identical geometry used by execution, without allocating a fake
    /// intermediate tensor during whole-network preflight.
    pub fn from_shapes(
        axes: &AxisPlan,
        weights_shape: &[usize],
        bias_shape: &[usize],
    ) -> Result<Self, DenseError> {
        use DeepLearningAxisLabel::{Batch as B, Channel as C, Time as T};
        if !matches!(
            axes.layout().axes(),
            [B, C] | [C, B] | [C, B, T] | [C, T] | [C]
        ) {
            return Err(DenseError::Invalid(
                "fullyConnectedLayer requires admitted feature or vector-sequence axes",
            ));
        }
        let channel_axis =
            axes.layout()
                .index(DeepLearningAxisLabel::Channel)
                .ok_or(DenseError::Invalid(
                    "fullyConnectedLayer needs a channel axis",
                ))?;
        let input_channels = axes.shape()[channel_axis];
        if input_channels == 0
            || weights_shape.len() != 2
            || weights_shape[0] == 0
            || weights_shape[1] != input_channels
        {
            return Err(DenseError::Invalid(
                "fullyConnectedLayer Weights must be output-by-input channels",
            ));
        }
        let output_channels = weights_shape[0];
        if bias_shape.len() > 2
            || bias_shape
                .iter()
                .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                != Some(output_channels)
        {
            return Err(DenseError::Invalid(
                "fullyConnectedLayer Bias must contain one value per output channel",
            ));
        }
        let output_axes = axes.with_channel_extent(output_channels)?;
        let input_strides = strides(axes.shape())?;
        let output_strides = strides(output_axes.shape())?;
        let mut sample_shape = Vec::new();
        let mut input_sample_strides = Vec::new();
        let mut output_sample_strides = Vec::new();
        let mut samples = 1usize;
        for axis in 0..axes.shape().len() {
            if axis == channel_axis {
                continue;
            }
            sample_shape.push(axes.shape()[axis]);
            input_sample_strides.push(input_strides[axis]);
            output_sample_strides.push(output_strides[axis]);
            samples = samples
                .checked_mul(axes.shape()[axis])
                .ok_or(DenseError::Overflow)?;
        }
        Ok(Self {
            output_axes,
            input_channels,
            output_channels,
            samples,
            input_channel_stride: input_strides[channel_axis],
            output_channel_stride: output_strides[channel_axis],
            sample_shape,
            input_sample_strides,
            output_sample_strides,
        })
    }

    pub fn bases(&self, mut sample: usize) -> (usize, usize) {
        let mut input = 0;
        let mut output = 0;
        for axis in 0..self.sample_shape.len() {
            let coordinate = sample % self.sample_shape[axis];
            sample /= self.sample_shape[axis];
            input += coordinate * self.input_sample_strides[axis];
            output += coordinate * self.output_sample_strides[axis];
        }
        (input, output)
    }

    pub fn input_offset(&self, base: usize, channel: usize) -> usize {
        base + channel * self.input_channel_stride
    }
    pub fn output_offset(&self, base: usize, channel: usize) -> usize {
        base + channel * self.output_channel_stride
    }
    pub fn weight_offset(&self, output: usize, input: usize) -> usize {
        output + input * self.output_channels
    }
}

pub(in crate::builtins::deep_learning) fn admit(
    axes: &AxisPlan,
    weights_shape: &[usize],
    bias_shape: &[usize],
) -> Result<AxisPlan, DenseError> {
    Ok(Geometry::from_shapes(axes, weights_shape, bias_shape)?.output_axes)
}
