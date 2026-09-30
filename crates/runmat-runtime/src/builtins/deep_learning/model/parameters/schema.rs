//! Typed identities and expected dimensions for the supported learnable roles.
use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum ParameterRole {
    Weights,
    Bias,
}

impl ParameterRole {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Weights => "Weights",
            Self::Bias => "Bias",
        }
    }

    pub(super) fn parse(name: &str) -> Option<Self> {
        match name {
            "Weights" => Some(Self::Weights),
            "Bias" => Some(Self::Bias),
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
pub(super) struct ParameterShape {
    pub(super) rows: usize,
    pub(super) columns: Option<usize>,
}

pub(super) struct ParameterView<'a> {
    pub(super) key: ParameterKey,
    pub(super) value: Option<&'a Value>,
    pub(super) expected_shape: ParameterShape,
}
