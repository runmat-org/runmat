use runmat_value::{ComplexTensor, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum StateRole {
    Hidden,
    Cell,
}

pub(in crate::builtins::deep_learning) struct StateUpdate {
    pub step_index: usize,
    pub role: StateRole,
    pub value: Tensor,
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

pub(super) struct StepRecord {
    pub step_index: usize,
    pub input: HostValue,
    pub input_axes: AxisPlan,
    pub output_axes: AxisPlan,
    pub cache: StepCache,
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
