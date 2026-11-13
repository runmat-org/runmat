//! Typed identities and expected dimensions for the supported learnable roles.
use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::builtins::deep_learning) enum ParameterRole {
    Weights,
    InputWeights,
    RecurrentWeights,
    Bias,
    Scale,
    Offset,
}

impl ParameterRole {
    pub(in crate::builtins::deep_learning) fn name(self) -> &'static str {
        match self {
            Self::Weights => "Weights",
            Self::InputWeights => "InputWeights",
            Self::RecurrentWeights => "RecurrentWeights",
            Self::Bias => "Bias",
            Self::Scale => "Scale",
            Self::Offset => "Offset",
        }
    }

    pub(in crate::builtins::deep_learning) fn parse(name: &str) -> Option<Self> {
        match name {
            "Weights" => Some(Self::Weights),
            "InputWeights" => Some(Self::InputWeights),
            "RecurrentWeights" => Some(Self::RecurrentWeights),
            "Bias" => Some(Self::Bias),
            "Scale" => Some(Self::Scale),
            "Offset" => Some(Self::Offset),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(in crate::builtins::deep_learning) struct ParameterKey {
    pub(in crate::builtins::deep_learning) layer_index: usize,
    pub(in crate::builtins::deep_learning) layer_name: String,
    pub(in crate::builtins::deep_learning) role: ParameterRole,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum ParameterShape {
    /// Matrix whose input width can remain unknown until an input is admitted.
    Matrix { rows: usize, columns: Option<usize> },
    /// An exact multidimensional geometry. An unknown channel dimension remains
    /// unknown rather than being guessed from a layer's current payload.
    Dimensions(Vec<Option<usize>>),
    /// The existing fully connected bias contract permits row or column form.
    Elements(usize),
}

pub(in crate::builtins::deep_learning) struct ParameterView<'a> {
    pub(in crate::builtins::deep_learning) key: ParameterKey,
    pub(in crate::builtins::deep_learning) value: Option<&'a Value>,
    pub(in crate::builtins::deep_learning) expected_shape: ParameterShape,
}
