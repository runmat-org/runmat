use runmat_builtins::{DeepLearningAxisLabel, DeepLearningAxisLayout};
use runmat_value::{NumericDType, Tensor};

use crate::builtins::deep_learning::model::plan::axes::{AxisGeometryError, AxisPlan};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum Activation {
    Tanh,
    Softsign,
    Relu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum GateActivation {
    Sigmoid,
    HardSigmoid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum OutputMode {
    Sequence,
    Last,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) struct RecurrentOptions {
    pub hidden_size: usize,
    pub bidirectional: bool,
    pub output_mode: OutputMode,
    pub state_activation: Activation,
    pub gate_activation: GateActivation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum RecurrentError {
    Invalid(&'static str),
    ShapeOverflow,
    UnsupportedPrecision,
    Axes(AxisGeometryError),
    Storage(String),
}

impl std::fmt::Display for RecurrentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
            Self::ShapeOverflow => formatter.write_str("recurrent geometry exceeds addressable size"),
            Self::UnsupportedPrecision => formatter.write_str(
                "recurrent input, parameters, state, and gradients require one matching real single or double class",
            ),
            Self::Axes(error) => write!(formatter, "recurrent axis geometry: {error:?}"),
            Self::Storage(error) => write!(formatter, "recurrent tensor storage: {error}"),
        }
    }
}

impl std::error::Error for RecurrentError {}

impl From<AxisGeometryError> for RecurrentError {
    fn from(error: AxisGeometryError) -> Self {
        Self::Axes(error)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Geometry {
    pub options: RecurrentOptions,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub output_axes: AxisPlan,
    pub input_channels: usize,
    pub batches: usize,
    pub steps: usize,
    pub directions: usize,
    pub state_batch: usize,
    input_strides: Vec<usize>,
    output_strides: Vec<usize>,
    channel_axis: Option<usize>,
    batch_axis: Option<usize>,
    time_axis: Option<usize>,
    spatial_axes: Vec<usize>,
    spatial_sizes: Vec<usize>,
}

fn strides(shape: &[usize]) -> Result<(Vec<usize>, usize), RecurrentError> {
    let mut next = 1usize;
    let mut strides = Vec::with_capacity(shape.len());
    for size in shape {
        strides.push(next);
        next = next
            .checked_mul(*size)
            .ok_or(RecurrentError::ShapeOverflow)?;
    }
    Ok((strides, next))
}

impl Geometry {
    pub fn new(
        input: &Tensor,
        input_weights: &Tensor,
        recurrent_weights: &Tensor,
        bias: &Tensor,
        initial_hidden: Option<&Tensor>,
        initial_cell: Option<&Tensor>,
        axes: &AxisPlan,
        options: RecurrentOptions,
    ) -> Result<Self, RecurrentError> {
        if axes.shape() != input.shape || options.hidden_size == 0 {
            return Err(RecurrentError::Invalid(
                "input axes must match shape and hidden size must be positive",
            ));
        }
        let dtype = input.numeric_dtype();
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64)
            || [input_weights, recurrent_weights, bias]
                .iter()
                .any(|tensor| tensor.numeric_dtype() != dtype)
            || [initial_hidden, initial_cell]
                .iter()
                .flatten()
                .any(|tensor| tensor.numeric_dtype() != dtype)
        {
            return Err(RecurrentError::UnsupportedPrecision);
        }
        let layout = axes.layout();
        let channel_axis = layout.index(DeepLearningAxisLabel::Channel);
        let batch_axis = layout.index(DeepLearningAxisLabel::Batch);
        let time_axis = layout.index(DeepLearningAxisLabel::Time);
        let spatial_axes = layout
            .axes()
            .iter()
            .enumerate()
            .filter_map(|(axis, label)| (*label == DeepLearningAxisLabel::Spatial).then_some(axis))
            .collect::<Vec<_>>();
        if channel_axis.is_none() && spatial_axes.is_empty() {
            return Err(RecurrentError::Invalid(
                "recurrent input needs a channel or spatial axis",
            ));
        }
        for (axis, label) in layout.axes().iter().enumerate() {
            if *label == DeepLearningAxisLabel::Unspecified && input.shape[axis] != 1 {
                return Err(RecurrentError::Invalid(
                    "non-singleton unspecified recurrent axes are unsupported",
                ));
            }
        }
        let spatial_sizes = spatial_axes
            .iter()
            .map(|axis| input.shape[*axis])
            .collect::<Vec<_>>();
        let input_channels = spatial_sizes
            .iter()
            .try_fold(
                channel_axis.map_or(1, |axis| input.shape[axis]),
                |n, size| n.checked_mul(*size),
            )
            .ok_or(RecurrentError::ShapeOverflow)?;
        let batches = batch_axis.map_or(1, |axis| input.shape[axis]);
        let steps = time_axis.map_or(1, |axis| input.shape[axis]);
        if input_channels == 0 || batches == 0 || steps == 0 {
            return Err(RecurrentError::Invalid(
                "recurrent channel, batch, and time dimensions must be nonempty",
            ));
        }
        let directions: usize = if options.bidirectional { 2 } else { 1 };
        let rows = options
            .hidden_size
            .checked_mul(4)
            .and_then(|n| n.checked_mul(directions))
            .ok_or(RecurrentError::ShapeOverflow)?;
        if input_weights.shape != [rows, input_channels]
            || recurrent_weights.shape != [rows, options.hidden_size]
            || bias.shape != [rows, 1]
        {
            return Err(RecurrentError::Invalid("InputWeights, RecurrentWeights, or Bias shape does not match recurrent gate geometry"));
        }
        if initial_hidden.is_some() != initial_cell.is_some() {
            return Err(RecurrentError::Invalid(
                "initial HiddenState and CellState must be supplied together",
            ));
        }
        let state_channels = directions
            .checked_mul(options.hidden_size)
            .ok_or(RecurrentError::ShapeOverflow)?;
        let state_batch = if let Some(hidden) = initial_hidden {
            let cell = initial_cell.expect("paired state checked above");
            let valid = |shape: &[usize]| match shape {
                [channels] => (*channels == state_channels).then_some(1),
                [channels, batch]
                    if *channels == state_channels && (*batch == 1 || *batch == batches) =>
                {
                    Some(*batch)
                }
                _ => None,
            };
            let hidden_batch = valid(&hidden.shape).ok_or(RecurrentError::Invalid(
                "HiddenState must be channels-by-batch or broadcast channels-by-one",
            ))?;
            let cell_batch = valid(&cell.shape).ok_or(RecurrentError::Invalid(
                "CellState must match HiddenState geometry",
            ))?;
            if hidden_batch != cell_batch {
                return Err(RecurrentError::Invalid(
                    "HiddenState and CellState batch dimensions differ",
                ));
            }
            hidden_batch
        } else {
            1
        };
        let output_channels = options
            .hidden_size
            .checked_mul(directions)
            .ok_or(RecurrentError::ShapeOverflow)?;
        let (output_shape, output_labels) = match (
            batch_axis.is_some(),
            time_axis.is_some(),
            options.output_mode,
        ) {
            (true, true, OutputMode::Sequence) => (vec![output_channels, batches, steps], "CBT"),
            (true, _, _) => (vec![output_channels, batches], "CB"),
            (false, true, OutputMode::Sequence) => (vec![output_channels, steps], "CT"),
            (false, _, _) => (vec![output_channels, 1], "CU"),
        };
        let output_layout = DeepLearningAxisLayout::parse(output_labels, output_shape.len())
            .map_err(|_| RecurrentError::Invalid("invalid recurrent output axes"))?;
        let output_axes = AxisPlan::new(output_layout, &output_shape)?;
        let (input_strides, input_elements) = strides(&input.shape)?;
        let (output_strides, _) = strides(&output_shape)?;
        if input_elements != input.len() {
            return Err(RecurrentError::Invalid(
                "input payload does not match its shape",
            ));
        }
        // Prove flat trace, gate, state, and output extents before any allocation.
        batches
            .checked_mul(steps)
            .and_then(|n| n.checked_mul(state_channels))
            .and_then(|n| n.checked_mul(9))
            .ok_or(RecurrentError::ShapeOverflow)?;
        rows.checked_mul(input_channels)
            .ok_or(RecurrentError::ShapeOverflow)?;
        rows.checked_mul(options.hidden_size)
            .ok_or(RecurrentError::ShapeOverflow)?;
        Ok(Self {
            options,
            input_shape: input.shape.clone(),
            output_shape,
            output_axes,
            input_channels,
            batches,
            steps,
            directions,
            state_batch,
            input_strides,
            output_strides,
            channel_axis,
            batch_axis,
            time_axis,
            spatial_axes,
            spatial_sizes,
        })
    }

    pub fn input_offset(&self, mut channel: usize, batch: usize, time: usize) -> usize {
        let mut offset = 0;
        for (axis, size) in self.spatial_axes.iter().zip(&self.spatial_sizes) {
            offset += (channel % *size) * self.input_strides[*axis];
            channel /= *size;
        }
        if let Some(axis) = self.channel_axis {
            offset += channel * self.input_strides[axis];
        }
        if let Some(axis) = self.batch_axis {
            offset += batch * self.input_strides[axis];
        }
        if let Some(axis) = self.time_axis {
            offset += time * self.input_strides[axis];
        }
        offset
    }

    pub fn output_offset(&self, channel: usize, batch: usize, time: usize) -> usize {
        channel * self.output_strides[0]
            + if self.batch_axis.is_some() {
                batch * self.output_strides[1]
            } else {
                0
            }
            + if self.time_axis.is_some() && self.options.output_mode == OutputMode::Sequence {
                time * self.output_strides[if self.batch_axis.is_some() { 2 } else { 1 }]
            } else {
                0
            }
    }

    pub fn state_offset(&self, direction: usize, hidden: usize, batch: usize) -> usize {
        direction * self.options.hidden_size
            + hidden
            + (if self.state_batch == 1 { 0 } else { batch })
                * self.options.hidden_size
                * self.directions
    }
}
