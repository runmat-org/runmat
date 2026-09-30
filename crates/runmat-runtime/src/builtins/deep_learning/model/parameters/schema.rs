//! Typed identities and expected dimensions for the supported learnable roles.
use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum ParameterRole {
    Weights,
    InputWeights,
    RecurrentWeights,
    Bias,
    Scale,
    Offset,
}

impl ParameterRole {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Weights => "Weights",
            Self::InputWeights => "InputWeights",
            Self::RecurrentWeights => "RecurrentWeights",
            Self::Bias => "Bias",
            Self::Scale => "Scale",
            Self::Offset => "Offset",
        }
    }

    pub(super) fn parse(name: &str) -> Option<Self> {
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
pub(super) struct ParameterKey {
    pub(super) layer_index: usize,
    pub(super) layer_name: String,
    pub(super) role: ParameterRole,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ParameterShape {
    /// Matrix whose input width can remain unknown until an input is admitted.
    Matrix { rows: usize, columns: Option<usize> },
    /// An exact multidimensional geometry. An unknown channel dimension remains
    /// unknown rather than being guessed from a layer's current payload.
    Dimensions(Vec<Option<usize>>),
    /// The existing fully connected bias contract permits row or column form.
    Elements(usize),
}

pub(super) struct ParameterView<'a> {
    pub(super) key: ParameterKey,
    pub(super) value: Option<&'a Value>,
    pub(super) expected_shape: ParameterShape,
}
