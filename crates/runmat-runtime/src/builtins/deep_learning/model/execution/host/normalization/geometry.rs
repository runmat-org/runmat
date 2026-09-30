use runmat_builtins::{DeepLearningAxisLabel, DeepLearningNormalizationAxes};
use runmat_value::{NumericDType, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::builtins::deep_learning) struct NormalizationOptions {
    pub epsilon: f64,
    pub operation: DeepLearningNormalizationAxes,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum NormalizationError {
    UnsupportedLayout,
    UnsupportedPrecision,
    InvalidEpsilon,
    EmptyReduction,
    ShapeMismatch,
    StatisticsMismatch,
    ShapeOverflow,
    Storage(String),
}

impl std::fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLayout => f.write_str("layerNormalizationLayer requires a supported feature, sequence, or 1-D spatial layout"),
            Self::UnsupportedPrecision => f.write_str("layerNormalizationLayer requires matching real single or double input, Scale, Offset, and gradient"),
            Self::InvalidEpsilon => f.write_str("layerNormalizationLayer Epsilon must remain positive and finite in input precision"),
            Self::EmptyReduction => f.write_str("layerNormalizationLayer cannot normalize an empty channel/spatial/time axis"),
            Self::ShapeMismatch => f.write_str("layerNormalizationLayer shape or channel parameter length mismatch"),
            Self::StatisticsMismatch => f.write_str("layerNormalizationLayer forward statistics do not match backward input"),
            Self::ShapeOverflow => f.write_str("layerNormalizationLayer geometry exceeds addressable size"),
            Self::Storage(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for NormalizationError {}

pub(super) struct Geometry {
    pub axes: AxisPlan,
    pub channels: usize,
    pub channel_axis: usize,
    pub reduce: Vec<bool>,
    pub group_strides: Vec<usize>,
    pub groups: usize,
    pub group_size: usize,
    pub epsilon: f64,
}

impl Geometry {
    pub fn new(axes: &AxisPlan, options: NormalizationOptions) -> Result<Self, NormalizationError> {
        if !options.epsilon.is_finite() || options.epsilon <= 0.0 {
            return Err(NormalizationError::InvalidEpsilon);
        }
        use DeepLearningAxisLabel::{Batch, Channel, Spatial, Time, Unspecified};
        if !matches!(
            axes.layout().axes(),
            [Channel]
                | [Batch, Channel]
                | [Channel, Batch]
                | [Channel, Time]
                | [Channel, Batch, Time]
                | [Spatial, Channel]
                | [Spatial, Channel, Batch]
                | [Spatial, Channel, Time]
                | [Spatial, Channel, Batch, Time]
                | [Channel, Unspecified]
        ) {
            return Err(NormalizationError::UnsupportedLayout);
        }
        let channel_axis = axes
            .layout()
            .index(DeepLearningAxisLabel::Channel)
            .ok_or(NormalizationError::UnsupportedLayout)?;
        let channels = axes.shape()[channel_axis];
        let reduce = axes
            .layout()
            .axes()
            .iter()
            .map(|axis| match options.operation {
                DeepLearningNormalizationAxes::Auto
                | DeepLearningNormalizationAxes::ChannelOnly => {
                    *axis == DeepLearningAxisLabel::Channel
                }
                DeepLearningNormalizationAxes::SpatialChannel => matches!(
                    axis,
                    DeepLearningAxisLabel::Channel | DeepLearningAxisLabel::Spatial
                ),
                DeepLearningNormalizationAxes::BatchExcluded => {
                    *axis != DeepLearningAxisLabel::Batch
                }
            })
            .collect::<Vec<_>>();
        let mut groups = 1usize;
        let mut group_size = 1usize;
        let mut group_strides = Vec::with_capacity(axes.shape().len());
        for (axis, &size) in axes.shape().iter().enumerate() {
            group_strides.push(if reduce[axis] { 0 } else { groups });
            if reduce[axis] {
                group_size = group_size
                    .checked_mul(size)
                    .ok_or(NormalizationError::ShapeOverflow)?;
            } else {
                groups = groups
                    .checked_mul(size)
                    .ok_or(NormalizationError::ShapeOverflow)?;
            }
        }
        if group_size == 0 || channels == 0 {
            return Err(NormalizationError::EmptyReduction);
        }
        Ok(Self {
            axes: axes.clone(),
            channels,
            channel_axis,
            reduce,
            group_strides,
            groups,
            group_size,
            epsilon: options.epsilon,
        })
    }

    pub fn validate_payload(
        &self,
        input: &Tensor,
        scale: &Tensor,
        offset: &Tensor,
    ) -> Result<(), NormalizationError> {
        if input.shape.as_slice() != self.axes.shape() || input.len() != self.axes.element_count() {
            return Err(NormalizationError::ShapeMismatch);
        }
        if scale.numeric_dtype() != input.numeric_dtype()
            || offset.numeric_dtype() != input.numeric_dtype()
        {
            return Err(NormalizationError::UnsupportedPrecision);
        }
        self.validate_shapes(&scale.shape, &offset.shape, input.numeric_dtype())
    }

    pub fn validate_shapes(
        &self,
        scale_shape: &[usize],
        offset_shape: &[usize],
        dtype: NumericDType,
    ) -> Result<(), NormalizationError> {
        // OperationDimension changes the reduction, not the learnable
        // parameters' channel-only geometry. Input kind selects the physical
        // orientation of the canonical Scale/Offset row or column.
        let parameter_shape = if self
            .axes
            .layout()
            .index(DeepLearningAxisLabel::Spatial)
            .is_some()
        {
            [1, self.channels]
        } else {
            [self.channels, 1]
        };
        if scale_shape != parameter_shape.as_slice() || offset_shape != parameter_shape.as_slice() {
            return Err(NormalizationError::ShapeMismatch);
        }
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
            return Err(NormalizationError::UnsupportedPrecision);
        }
        if dtype == NumericDType::F32
            && (self.epsilon as f32 == 0.0 || !(self.epsilon as f32).is_finite())
        {
            return Err(NormalizationError::InvalidEpsilon);
        }
        Ok(())
    }

    pub fn group_and_channel(&self, mut linear: usize) -> (usize, usize) {
        let mut group = 0usize;
        let mut channel = 0usize;
        for (axis, &extent) in self.axes.shape().iter().enumerate() {
            let coordinate = linear % extent;
            linear /= extent;
            group += coordinate * self.group_strides[axis];
            if axis == self.channel_axis {
                channel = coordinate;
            }
        }
        (group, channel)
    }
}
