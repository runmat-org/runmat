use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{ComplexTensor, NumericDType, Tensor};

use crate::builtins::deep_learning::model::plan::axes::{AxisGeometryError, AxisPlan};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum PoolError {
    UnsupportedLayout,
    UnsupportedPrecision,
    EmptyPool,
    ShapeMismatch,
    Axes(AxisGeometryError),
    Storage(String),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLayout => f.write_str(
                "globalAveragePooling1dLayer requires a supported spatial or time axis layout",
            ),
            Self::UnsupportedPrecision => f.write_str(
                "globalAveragePooling1dLayer requires single or double real or complex data",
            ),
            Self::EmptyPool => f.write_str(
                "globalAveragePooling1dLayer cannot average an empty spatial or time axis",
            ),
            Self::ShapeMismatch => {
                f.write_str("globalAveragePooling1dLayer payload/gradient shape mismatch")
            }
            Self::Axes(error) => write!(f, "pooling axis geometry: {error:?}"),
            Self::Storage(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for PoolError {}

pub(super) struct Geometry {
    pub input: AxisPlan,
    pub output: AxisPlan,
    pub pool_axis: usize,
    pub pool_size: usize,
    pool_stride: usize,
    input_strides: Vec<usize>,
}

impl Geometry {
    pub fn new(axes: &AxisPlan) -> Result<Self, PoolError> {
        use DeepLearningAxisLabel::{Batch, Channel, Spatial, Time};
        if !matches!(
            axes.layout().axes(),
            [Channel, Time]
                | [Channel, Batch, Time]
                | [Spatial, Channel]
                | [Spatial, Channel, Batch]
                | [Spatial, Channel, Time]
                | [Spatial, Channel, Batch, Time]
                | [Batch, Time]
                | [Spatial, Batch]
                | [Spatial, Batch, Time]
        ) {
            return Err(PoolError::UnsupportedLayout);
        }
        let pool_axis = axes
            .layout()
            .index(DeepLearningAxisLabel::Spatial)
            .or_else(|| axes.layout().index(DeepLearningAxisLabel::Time))
            .ok_or(PoolError::UnsupportedLayout)?;
        let pool_size = axes.shape()[pool_axis];
        if pool_size == 0 {
            return Err(PoolError::EmptyPool);
        }
        let output = axes
            .after_global_average_pooling_1d()
            .map_err(PoolError::Axes)?;
        let mut stride = 1usize;
        let mut input_strides = Vec::with_capacity(axes.shape().len());
        for &extent in axes.shape() {
            input_strides.push(stride);
            stride = stride
                .checked_mul(extent)
                .ok_or(PoolError::Axes(AxisGeometryError::ShapeOverflow))?;
        }
        let pool_stride = input_strides[pool_axis];
        Ok(Self {
            input: axes.clone(),
            output,
            pool_axis,
            pool_size,
            pool_stride,
            input_strides,
        })
    }

    pub fn validate_input(&self, input: &Tensor) -> Result<(), PoolError> {
        if input.shape() != self.input.shape() || input.len() != self.input.element_count() {
            return Err(PoolError::ShapeMismatch);
        }
        if !matches!(input.numeric_dtype(), NumericDType::F32 | NumericDType::F64) {
            return Err(PoolError::UnsupportedPrecision);
        }
        Ok(())
    }

    pub fn validate_complex_input(&self, input: &ComplexTensor) -> Result<(), PoolError> {
        if input.shape() != self.input.shape() || input.len() != self.input.element_count() {
            return Err(PoolError::ShapeMismatch);
        }
        if !matches!(input.numeric_dtype(), NumericDType::F32 | NumericDType::F64) {
            return Err(PoolError::UnsupportedPrecision);
        }
        Ok(())
    }

    pub fn input_base_offset(&self, output_linear: usize) -> usize {
        let mut linear = output_linear;
        let mut base = 0;
        for (output_axis, &size) in self.output.shape().iter().enumerate() {
            let coordinate = linear % size;
            linear /= size;
            let input_axis = if self.output.shape().len() == self.input.shape().len()
                || output_axis < self.pool_axis
            {
                output_axis
            } else {
                output_axis + 1
            };
            if input_axis != self.pool_axis {
                base += coordinate * self.input_strides[input_axis];
            }
        }
        base
    }

    pub fn input_offset(&self, base: usize, pool_index: usize) -> usize {
        base + pool_index * self.pool_stride
    }
}
