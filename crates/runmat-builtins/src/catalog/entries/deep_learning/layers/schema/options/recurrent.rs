use super::DeepLearningLayerOption as O;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningRecurrentOutputMode {
    Sequence,
    Last,
}

impl DeepLearningRecurrentOutputMode {
    pub const ALL: [Self; 2] = [Self::Sequence, Self::Last];
    pub const NAMES: &'static [&'static str] =
        &[Self::Sequence.canonical_name(), Self::Last.canonical_name()];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| name.eq_ignore_ascii_case(mode.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Sequence => "sequence",
            Self::Last => "last",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningRecurrentStateActivation {
    Tanh,
    Softsign,
    Relu,
}

impl DeepLearningRecurrentStateActivation {
    pub const ALL: [Self; 3] = [Self::Tanh, Self::Softsign, Self::Relu];
    pub const NAMES: &'static [&'static str] = &[
        Self::Tanh.canonical_name(),
        Self::Softsign.canonical_name(),
        Self::Relu.canonical_name(),
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| name.eq_ignore_ascii_case(mode.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::Softsign => "softsign",
            Self::Relu => "relu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningRecurrentGateActivation {
    Sigmoid,
    HardSigmoid,
}

impl DeepLearningRecurrentGateActivation {
    pub const ALL: [Self; 2] = [Self::Sigmoid, Self::HardSigmoid];
    pub const NAMES: &'static [&'static str] = &[
        Self::Sigmoid.canonical_name(),
        Self::HardSigmoid.canonical_name(),
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| name.eq_ignore_ascii_case(mode.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::HardSigmoid => "hard-sigmoid",
        }
    }
}

pub(super) const fn supports(option: O) -> bool {
    matches!(
        option,
        O::OutputMode
            | O::HasStateInputs
            | O::HasStateOutputs
            | O::InputSize
            | O::StateActivationFunction
            | O::GateActivationFunction
            | O::CellState
            | O::HiddenState
            | O::InputWeightsInitializer
            | O::RecurrentWeightsInitializer
            | O::BiasInitializer
            | O::InputWeights
            | O::RecurrentWeights
            | O::Bias
            | O::InputWeightsLearnRateFactor
            | O::RecurrentWeightsLearnRateFactor
            | O::BiasLearnRateFactor
            | O::InputWeightsL2Factor
            | O::RecurrentWeightsL2Factor
            | O::BiasL2Factor
    )
}
