use runmat_builtins::{
    DeepLearningAxisLabel, DeepLearningConvolutionPaddingMode,
    DeepLearningConvolutionPaddingValueMode,
};
use runmat_value::{NumericDType, Tensor};

use crate::builtins::deep_learning::model::plan::axes::{AxisGeometryError, AxisPlan};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::builtins::deep_learning) enum Padding {
    Valid,
    Same,
    Causal,
    Explicit { left: usize, right: usize },
}

impl From<DeepLearningConvolutionPaddingMode> for Padding {
    fn from(mode: DeepLearningConvolutionPaddingMode) -> Self {
        match mode {
            DeepLearningConvolutionPaddingMode::Same => Self::Same,
            DeepLearningConvolutionPaddingMode::Causal => Self::Causal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::builtins::deep_learning) enum PaddingValue {
    Constant(f64),
    Replicate,
    SymmetricIncludeEdge,
    SymmetricExcludeEdge,
}

impl From<DeepLearningConvolutionPaddingValueMode> for PaddingValue {
    fn from(mode: DeepLearningConvolutionPaddingValueMode) -> Self {
        match mode {
            DeepLearningConvolutionPaddingValueMode::Replicate => Self::Replicate,
            DeepLearningConvolutionPaddingValueMode::SymmetricIncludeEdge => {
                Self::SymmetricIncludeEdge
            }
            DeepLearningConvolutionPaddingValueMode::SymmetricExcludeEdge => {
                Self::SymmetricExcludeEdge
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::builtins::deep_learning) struct ConvOptions {
    pub stride: usize,
    pub dilation: usize,
    pub padding: Padding,
    pub padding_value: PaddingValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum ConvError {
    Invalid(&'static str),
    ShapeOverflow,
    UnsupportedPrecision,
    UnsupportedLayout,
    Axes(AxisGeometryError),
    Storage(String),
}

impl std::fmt::Display for ConvError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
            Self::ShapeOverflow => formatter.write_str("convolution geometry exceeds addressable size"),
            Self::UnsupportedPrecision => formatter.write_str("convolution requires matching real single or double input, weights, bias, and gradient"),
            Self::UnsupportedLayout => formatter.write_str("convolution requires CBT, CT, SCB, SC, SCBT, or SCT axes"),
            Self::Axes(error) => write!(formatter, "convolution axis geometry: {error:?}"),
            Self::Storage(error) => write!(formatter, "convolution tensor storage: {error}"),
        }
    }
}

impl std::error::Error for ConvError {}

impl From<AxisGeometryError> for ConvError {
    fn from(error: AxisGeometryError) -> Self {
        Self::Axes(error)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Geometry {
    pub output_shape: Vec<usize>,
    pub output_elements: usize,
    pub input_strides: Vec<usize>,
    pub output_strides: Vec<usize>,
    pub span_axis: usize,
    pub channel_axis: usize,
    pub batch_axis: Option<usize>,
    pub time_axis: Option<usize>,
    pub input_len: usize,
    pub output_len: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub filter_size: usize,
    pub batch_size: usize,
    pub time_size: usize,
    pub left: usize,
    pub right: usize,
    pub options: ConvOptions,
}

fn strides(shape: &[usize]) -> Result<(Vec<usize>, usize), ConvError> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut elements = 1usize;
    for &size in shape {
        strides.push(elements);
        elements = elements.checked_mul(size).ok_or(ConvError::ShapeOverflow)?;
    }
    Ok((strides, elements))
}

impl Geometry {
    pub fn new(
        input: &Tensor,
        weights: &Tensor,
        bias: &Tensor,
        axes: &AxisPlan,
        options: ConvOptions,
    ) -> Result<Self, ConvError> {
        if axes.shape() != input.shape() || input.len() != axes.element_count() {
            return Err(ConvError::Invalid(
                "input axes and payload must match the input",
            ));
        }
        if !matches!(input.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
            || input.numeric_dtype() != weights.numeric_dtype()
            || input.numeric_dtype() != bias.numeric_dtype()
        {
            return Err(ConvError::UnsupportedPrecision);
        }
        Self::from_shapes(
            axes,
            weights.shape(),
            bias.shape(),
            input.numeric_dtype(),
            options,
        )
    }

    pub fn from_shapes(
        axes: &AxisPlan,
        weights_shape: &[usize],
        bias_shape: &[usize],
        dtype: NumericDType,
        options: ConvOptions,
    ) -> Result<Self, ConvError> {
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64)
            || options.stride == 0
            || options.dilation == 0
        {
            return Err(ConvError::Invalid(
                "positive stride/dilation and floating precision are required",
            ));
        }
        use DeepLearningAxisLabel::{Batch, Channel, Spatial, Time};
        if !matches!(
            axes.layout().axes(),
            [Channel, Batch, Time]
                | [Channel, Time]
                | [Spatial, Channel, Batch]
                | [Spatial, Channel]
                | [Spatial, Channel, Batch, Time]
                | [Spatial, Channel, Time]
        ) {
            return Err(ConvError::UnsupportedLayout);
        }
        let channel_axis = axes
            .layout()
            .index(DeepLearningAxisLabel::Channel)
            .ok_or(ConvError::UnsupportedLayout)?;
        let span_axis = axes
            .layout()
            .index(DeepLearningAxisLabel::Spatial)
            .or_else(|| axes.layout().index(DeepLearningAxisLabel::Time))
            .ok_or(ConvError::UnsupportedLayout)?;
        let batch_axis = axes.layout().index(DeepLearningAxisLabel::Batch);
        let time_axis = axes
            .layout()
            .index(DeepLearningAxisLabel::Spatial)
            .and_then(|_| axes.layout().index(DeepLearningAxisLabel::Time));
        let input_len = axes.shape()[span_axis];
        let input_channels = axes.shape()[channel_axis];
        if input_len == 0 || input_channels == 0 {
            return Err(ConvError::Invalid(
                "convolution requires a nonempty spatial/time and channel axis",
            ));
        }
        if weights_shape.len() != 3
            || weights_shape[0] == 0
            || weights_shape[1] != input_channels
            || weights_shape[2] == 0
        {
            return Err(ConvError::Invalid(
                "Weights must be FilterSize-by-NumChannels-by-NumFilters",
            ));
        }
        let filter_size = weights_shape[0];
        let output_channels = weights_shape[2];
        if bias_shape != [1, output_channels] {
            return Err(ConvError::Invalid("Bias must be 1-by-NumFilters"));
        }
        if let PaddingValue::Constant(value) = options.padding_value {
            if !value.is_finite() || (dtype == NumericDType::F32 && !(value as f32).is_finite()) {
                return Err(ConvError::Invalid("PaddingValue must be finite"));
            }
        }
        let effective_filter = filter_size
            .checked_sub(1)
            .and_then(|count| count.checked_mul(options.dilation))
            .and_then(|count| count.checked_add(1))
            .ok_or(ConvError::ShapeOverflow)?;
        let (left, right, output_len) = match options.padding {
            Padding::Valid => (
                0,
                0,
                output_size(input_len, 0, 0, effective_filter, options.stride)?,
            ),
            Padding::Explicit { left, right } => (
                left,
                right,
                output_size(input_len, left, right, effective_filter, options.stride)?,
            ),
            Padding::Causal => {
                let left = effective_filter - 1;
                (
                    left,
                    0,
                    output_size(input_len, left, 0, effective_filter, options.stride)?,
                )
            }
            Padding::Same => {
                let output_len = input_len / options.stride
                    + usize::from(!input_len.is_multiple_of(options.stride));
                let required = (output_len - 1)
                    .checked_mul(options.stride)
                    .and_then(|start| start.checked_add(effective_filter))
                    .ok_or(ConvError::ShapeOverflow)?;
                let total = required.saturating_sub(input_len);
                let left = total / 2;
                (left, total - left, output_len)
            }
        };
        if matches!(options.padding_value, PaddingValue::SymmetricExcludeEdge)
            && input_len == 1
            && (left != 0 || right != 0)
        {
            return Err(ConvError::Invalid(
                "symmetric-exclude-edge padding requires at least two samples",
            ));
        }
        // Prove every loop coordinate is representable before any allocation.
        if output_len > 0 {
            (output_len - 1)
                .checked_mul(options.stride)
                .and_then(|start| start.checked_add(effective_filter - 1))
                .ok_or(ConvError::ShapeOverflow)?;
        }
        let mut output_shape = axes.shape().to_vec();
        output_shape[span_axis] = output_len;
        output_shape[channel_axis] = output_channels;
        let (input_strides, input_elements) = strides(axes.shape())?;
        let (output_strides, output_elements) = strides(&output_shape)?;
        if input_elements != axes.element_count() {
            return Err(ConvError::Invalid("input payload does not match its shape"));
        }
        Ok(Self {
            output_shape,
            output_elements,
            input_strides,
            output_strides,
            span_axis,
            channel_axis,
            batch_axis,
            time_axis,
            input_len,
            output_len,
            input_channels,
            output_channels,
            filter_size,
            batch_size: batch_axis.map_or(1, |axis| axes.shape()[axis]),
            time_size: time_axis.map_or(1, |axis| axes.shape()[axis]),
            left,
            right,
            options,
        })
    }

    pub fn input_offset(&self, span: usize, channel: usize, batch: usize, time: usize) -> usize {
        span * self.input_strides[self.span_axis]
            + channel * self.input_strides[self.channel_axis]
            + self
                .batch_axis
                .map_or(0, |axis| batch * self.input_strides[axis])
            + self
                .time_axis
                .map_or(0, |axis| time * self.input_strides[axis])
    }

    pub fn output_offset(&self, span: usize, channel: usize, batch: usize, time: usize) -> usize {
        span * self.output_strides[self.span_axis]
            + channel * self.output_strides[self.channel_axis]
            + self
                .batch_axis
                .map_or(0, |axis| batch * self.output_strides[axis])
            + self
                .time_axis
                .map_or(0, |axis| time * self.output_strides[axis])
    }

    pub fn weight_offset(
        &self,
        filter: usize,
        input_channel: usize,
        output_channel: usize,
    ) -> usize {
        filter
            + input_channel * self.filter_size
            + output_channel * self.filter_size * self.input_channels
    }

    pub fn mapped_span(&self, output_span: usize, filter: usize) -> Option<usize> {
        let position = output_span * self.options.stride + filter * self.options.dilation;
        let coordinate = position as i128 - self.left as i128;
        let len = self.input_len as i128;
        if (0..len).contains(&coordinate) {
            return Some(coordinate as usize);
        }
        match self.options.padding_value {
            PaddingValue::Constant(_) => None,
            PaddingValue::Replicate => Some(if coordinate < 0 {
                0
            } else {
                self.input_len - 1
            }),
            PaddingValue::SymmetricIncludeEdge => {
                let period = len * 2;
                let index = coordinate.rem_euclid(period);
                Some(if index < len {
                    index
                } else {
                    period - index - 1
                } as usize)
            }
            PaddingValue::SymmetricExcludeEdge => {
                if self.input_len == 1 {
                    return Some(0);
                }
                let period = (len - 1) * 2;
                let index = coordinate.rem_euclid(period);
                Some(if index < len { index } else { period - index } as usize)
            }
        }
    }
}

fn output_size(
    input: usize,
    left: usize,
    right: usize,
    effective_filter: usize,
    stride: usize,
) -> Result<usize, ConvError> {
    let padded = input
        .checked_add(left)
        .and_then(|v| v.checked_add(right))
        .ok_or(ConvError::ShapeOverflow)?;
    if padded < effective_filter {
        Ok(0)
    } else {
        Ok((padded - effective_filter) / stride + 1)
    }
}
