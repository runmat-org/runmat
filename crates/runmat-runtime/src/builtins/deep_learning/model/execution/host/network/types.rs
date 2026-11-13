use runmat_value::{ComplexTensor, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use crate::builtins::deep_learning::model::state::StateUpdate;

use super::super::{activation, conv, dense, normalization, pooling, recurrent, softmax};

pub(in crate::builtins::deep_learning) enum HostValue {
    Real(Tensor),
    Complex(ComplexTensor),
}

pub(in crate::builtins::deep_learning) enum HostStep<'a> {
    Input,
    FullyConnected {
        weights: &'a Tensor,
        bias: &'a Tensor,
    },
    Relu,
    Elu {
        alpha: f64,
    },
    Softmax,
    Convolution1d {
        weights: &'a Tensor,
        bias: &'a Tensor,
        options: conv::ConvOptions,
    },
    LayerNormalization {
        scale: &'a Tensor,
        offset: &'a Tensor,
        options: normalization::NormalizationOptions,
    },
    GlobalAveragePooling1d,
    Recurrent {
        input_weights: &'a Tensor,
        recurrent_weights: &'a Tensor,
        bias: &'a Tensor,
        initial_hidden: Option<&'a Tensor>,
        initial_cell: Option<&'a Tensor>,
        options: recurrent::RecurrentOptions,
    },
    ClassificationOutput,
    RegressionOutput,
}

pub(in crate::builtins::deep_learning) struct HostExecutionResult {
    pub value: HostValue,
    pub axes: AxisPlan,
    pub trace: Option<HostTrace>,
    pub next_state: Vec<StateUpdate>,
}

pub(in crate::builtins::deep_learning) struct HostTrace {
    pub(super) records: Vec<StepRecord>,
    pub(super) output_axes: AxisPlan,
}

impl HostTrace {
    pub(in crate::builtins::deep_learning) fn output_axes(&self) -> &AxisPlan {
        &self.output_axes
    }
}

pub(super) struct StepRecord {
    pub step_index: usize,
    pub step: PinnedStep,
    pub input: HostValue,
    pub input_axes: AxisPlan,
    pub output_axes: AxisPlan,
    pub cache: StepCache,
}

/// Arc/CoW-backed immutable forward parameter snapshot. This is a tape
/// lifetime pin, not a second mutable network parameter authority.
pub(super) enum PinnedStep {
    FullyConnected {
        weights: Tensor,
        bias: Tensor,
    },
    Relu,
    Elu {
        alpha: f64,
    },
    Softmax,
    Convolution1d {
        weights: Tensor,
        bias: Tensor,
        options: conv::ConvOptions,
    },
    LayerNormalization {
        scale: Tensor,
        offset: Tensor,
        options: normalization::NormalizationOptions,
    },
    GlobalAveragePooling1d,
    Recurrent {
        input_weights: Tensor,
        recurrent_weights: Tensor,
        bias: Tensor,
        initial_hidden: Option<Tensor>,
        initial_cell: Option<Tensor>,
        options: recurrent::RecurrentOptions,
    },
}

impl PinnedStep {
    pub fn capture(step: &HostStep<'_>) -> Option<Self> {
        Some(match step {
            HostStep::Input | HostStep::ClassificationOutput | HostStep::RegressionOutput => {
                return None
            }
            HostStep::FullyConnected { weights, bias } => Self::FullyConnected {
                weights: (**weights).clone(),
                bias: (**bias).clone(),
            },
            HostStep::Relu => Self::Relu,
            HostStep::Elu { alpha } => Self::Elu { alpha: *alpha },
            HostStep::Softmax => Self::Softmax,
            HostStep::Convolution1d {
                weights,
                bias,
                options,
            } => Self::Convolution1d {
                weights: (**weights).clone(),
                bias: (**bias).clone(),
                options: *options,
            },
            HostStep::LayerNormalization {
                scale,
                offset,
                options,
            } => Self::LayerNormalization {
                scale: (**scale).clone(),
                offset: (**offset).clone(),
                options: *options,
            },
            HostStep::GlobalAveragePooling1d => Self::GlobalAveragePooling1d,
            HostStep::Recurrent {
                input_weights,
                recurrent_weights,
                bias,
                initial_hidden,
                initial_cell,
                options,
            } => Self::Recurrent {
                input_weights: (**input_weights).clone(),
                recurrent_weights: (**recurrent_weights).clone(),
                bias: (**bias).clone(),
                initial_hidden: initial_hidden.map(|tensor| (*tensor).clone()),
                initial_cell: initial_cell.map(|tensor| (*tensor).clone()),
                options: *options,
            },
        })
    }
}

pub(super) enum StepCache {
    None,
    Softmax(Tensor),
    Normalization(normalization::NormalizationStats),
    Recurrent(recurrent::RecurrentBackwardCache),
}

#[derive(Debug)]
pub(in crate::builtins::deep_learning) enum HostError {
    Invalid(&'static str),
    TraceMismatch,
    Dense(dense::DenseError),
    Activation(activation::ActivationError),
    Softmax(softmax::SoftmaxError),
    Convolution(conv::ConvError),
    Normalization(normalization::NormalizationError),
    Pooling(pooling::PoolError),
    Recurrent(recurrent::RecurrentError),
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::TraceMismatch => {
                f.write_str("host network reverse trace differs from admitted forward execution")
            }
            Self::Dense(error) => error.fmt(f),
            Self::Activation(error) => error.fmt(f),
            Self::Softmax(error) => error.fmt(f),
            Self::Convolution(error) => error.fmt(f),
            Self::Normalization(error) => error.fmt(f),
            Self::Pooling(error) => error.fmt(f),
            Self::Recurrent(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for HostError {}

impl From<dense::DenseError> for HostError {
    fn from(error: dense::DenseError) -> Self {
        Self::Dense(error)
    }
}
impl From<activation::ActivationError> for HostError {
    fn from(error: activation::ActivationError) -> Self {
        Self::Activation(error)
    }
}
impl From<softmax::SoftmaxError> for HostError {
    fn from(error: softmax::SoftmaxError) -> Self {
        Self::Softmax(error)
    }
}
impl From<conv::ConvError> for HostError {
    fn from(error: conv::ConvError) -> Self {
        Self::Convolution(error)
    }
}
impl From<normalization::NormalizationError> for HostError {
    fn from(error: normalization::NormalizationError) -> Self {
        Self::Normalization(error)
    }
}
impl From<pooling::PoolError> for HostError {
    fn from(error: pooling::PoolError) -> Self {
        Self::Pooling(error)
    }
}
impl From<recurrent::RecurrentError> for HostError {
    fn from(error: recurrent::RecurrentError) -> Self {
        Self::Recurrent(error)
    }
}
