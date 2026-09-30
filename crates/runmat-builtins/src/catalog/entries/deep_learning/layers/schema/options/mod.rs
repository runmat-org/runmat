//! Typed name-value identities. Text is interpreted only once at the public
//! constructor boundary; plans and executors use these closed identities.

mod convolution;
mod dense;
mod input;
mod normalization;
mod output;
mod policy;
mod recurrent;

pub use convolution::{
    DeepLearningConvolutionPaddingMode, DeepLearningConvolutionPaddingValueMode,
};
pub use dense::DeepLearningDenseOperationDimension;
pub use normalization::DeepLearningNormalizationAxes;
pub use recurrent::{
    DeepLearningRecurrentGateActivation, DeepLearningRecurrentOutputMode,
    DeepLearningRecurrentStateActivation,
};

use super::DeepLearningLayerKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum DeepLearningLayerOption {
    Name,
    Description,
    Normalization,
    NormalizationDimension,
    Mean,
    StandardDeviation,
    Min,
    Max,
    SplitComplexInputs,
    MinLength,
    InputSize,
    OperationDimension,
    WeightsInitializer,
    BiasInitializer,
    Weights,
    Bias,
    WeightLearnRateFactor,
    WeightL2Factor,
    BiasLearnRateFactor,
    BiasL2Factor,
    InputLearnables,
    OutputLearnables,
    Stride,
    DilationFactor,
    Padding,
    PaddingValue,
    NumChannels,
    OutputMode,
    HasStateInputs,
    HasStateOutputs,
    StateActivationFunction,
    GateActivationFunction,
    CellState,
    HiddenState,
    InputWeightsInitializer,
    RecurrentWeightsInitializer,
    InputWeights,
    RecurrentWeights,
    InputWeightsLearnRateFactor,
    RecurrentWeightsLearnRateFactor,
    InputWeightsL2Factor,
    RecurrentWeightsL2Factor,
    Epsilon,
    ScaleInitializer,
    OffsetInitializer,
    Scale,
    Offset,
    ScaleLearnRateFactor,
    OffsetLearnRateFactor,
    ScaleL2Factor,
    OffsetL2Factor,
    Classes,
    ClassWeights,
    ResponseNames,
}

/// Constructor acceptance and downstream executable-plan status are separate.
/// `PendingExecution` may be stored canonically, but a plan must not ignore a
/// nondefault operational value. The numerical layer work must close each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerOptionExecution {
    Metadata,
    Executable,
    PendingExecution,
}

/// A declared constructor default, not an assertion that an execution kernel
/// consumes the option. `None` means the schema has no proven inert default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerOptionDefault {
    Text(&'static str),
    Number(f64),
    Bool(bool),
}

impl DeepLearningLayerOption {
    pub const fn is_recurrent_gate_factor(self) -> bool {
        use DeepLearningLayerOption as O;
        matches!(
            self,
            O::BiasLearnRateFactor
                | O::BiasL2Factor
                | O::InputWeightsLearnRateFactor
                | O::RecurrentWeightsLearnRateFactor
                | O::InputWeightsL2Factor
                | O::RecurrentWeightsL2Factor
        )
    }

    pub fn from_name(name: &str) -> Option<Self> {
        use DeepLearningLayerOption as O;
        Some(match name.to_ascii_lowercase().as_str() {
            "name" => O::Name,
            "description" => O::Description,
            "normalization" => O::Normalization,
            "normalizationdimension" => O::NormalizationDimension,
            "mean" => O::Mean,
            "standarddeviation" => O::StandardDeviation,
            "min" => O::Min,
            "max" => O::Max,
            "splitcomplexinputs" => O::SplitComplexInputs,
            "minlength" => O::MinLength,
            "inputsize" => O::InputSize,
            "operationdimension" => O::OperationDimension,
            "weightsinitializer" => O::WeightsInitializer,
            "biasinitializer" => O::BiasInitializer,
            "weights" => O::Weights,
            "bias" => O::Bias,
            // Earlier RunMat constructors admitted these plural spellings.
            "weightlearnratefactor" | "weightslearnratefactor" => O::WeightLearnRateFactor,
            "weightl2factor" | "weightsl2factor" => O::WeightL2Factor,
            "biaslearnratefactor" => O::BiasLearnRateFactor,
            "biasl2factor" => O::BiasL2Factor,
            "inputlearnables" => O::InputLearnables,
            "outputlearnables" => O::OutputLearnables,
            "stride" => O::Stride,
            "dilationfactor" => O::DilationFactor,
            "padding" => O::Padding,
            "paddingvalue" => O::PaddingValue,
            "numchannels" => O::NumChannels,
            "outputmode" => O::OutputMode,
            "hasstateinputs" => O::HasStateInputs,
            "hasstateoutputs" => O::HasStateOutputs,
            "stateactivationfunction" => O::StateActivationFunction,
            "gateactivationfunction" => O::GateActivationFunction,
            "cellstate" => O::CellState,
            "hiddenstate" => O::HiddenState,
            "inputweightsinitializer" => O::InputWeightsInitializer,
            "recurrentweightsinitializer" => O::RecurrentWeightsInitializer,
            "inputweights" => O::InputWeights,
            "recurrentweights" => O::RecurrentWeights,
            "inputweightslearnratefactor" => O::InputWeightsLearnRateFactor,
            "recurrentweightslearnratefactor" => O::RecurrentWeightsLearnRateFactor,
            "inputweightsl2factor" => O::InputWeightsL2Factor,
            "recurrentweightsl2factor" => O::RecurrentWeightsL2Factor,
            "epsilon" => O::Epsilon,
            "scaleinitializer" => O::ScaleInitializer,
            "offsetinitializer" => O::OffsetInitializer,
            "scale" => O::Scale,
            "offset" => O::Offset,
            "scalelearnratefactor" => O::ScaleLearnRateFactor,
            "offsetlearnratefactor" => O::OffsetLearnRateFactor,
            "scalel2factor" => O::ScaleL2Factor,
            "offsetl2factor" => O::OffsetL2Factor,
            "classes" => O::Classes,
            "classweights" => O::ClassWeights,
            "responsenames" => O::ResponseNames,
            _ => return None,
        })
    }

    pub const fn canonical_name(self) -> &'static str {
        use DeepLearningLayerOption as O;
        match self {
            O::Name => "Name",
            O::Description => "Description",
            O::Normalization => "Normalization",
            O::NormalizationDimension => "NormalizationDimension",
            O::Mean => "Mean",
            O::StandardDeviation => "StandardDeviation",
            O::Min => "Min",
            O::Max => "Max",
            O::SplitComplexInputs => "SplitComplexInputs",
            O::MinLength => "MinLength",
            O::InputSize => "InputSize",
            O::OperationDimension => "OperationDimension",
            O::WeightsInitializer => "WeightsInitializer",
            O::BiasInitializer => "BiasInitializer",
            O::Weights => "Weights",
            O::Bias => "Bias",
            O::WeightLearnRateFactor => "WeightLearnRateFactor",
            O::WeightL2Factor => "WeightL2Factor",
            O::BiasLearnRateFactor => "BiasLearnRateFactor",
            O::BiasL2Factor => "BiasL2Factor",
            O::InputLearnables => "InputLearnables",
            O::OutputLearnables => "OutputLearnables",
            O::Stride => "Stride",
            O::DilationFactor => "DilationFactor",
            O::Padding => "Padding",
            O::PaddingValue => "PaddingValue",
            O::NumChannels => "NumChannels",
            O::OutputMode => "OutputMode",
            O::HasStateInputs => "HasStateInputs",
            O::HasStateOutputs => "HasStateOutputs",
            O::StateActivationFunction => "StateActivationFunction",
            O::GateActivationFunction => "GateActivationFunction",
            O::CellState => "CellState",
            O::HiddenState => "HiddenState",
            O::InputWeightsInitializer => "InputWeightsInitializer",
            O::RecurrentWeightsInitializer => "RecurrentWeightsInitializer",
            O::InputWeights => "InputWeights",
            O::RecurrentWeights => "RecurrentWeights",
            O::InputWeightsLearnRateFactor => "InputWeightsLearnRateFactor",
            O::RecurrentWeightsLearnRateFactor => "RecurrentWeightsLearnRateFactor",
            O::InputWeightsL2Factor => "InputWeightsL2Factor",
            O::RecurrentWeightsL2Factor => "RecurrentWeightsL2Factor",
            O::Epsilon => "Epsilon",
            O::ScaleInitializer => "ScaleInitializer",
            O::OffsetInitializer => "OffsetInitializer",
            O::Scale => "Scale",
            O::Offset => "Offset",
            O::ScaleLearnRateFactor => "ScaleLearnRateFactor",
            O::OffsetLearnRateFactor => "OffsetLearnRateFactor",
            O::ScaleL2Factor => "ScaleL2Factor",
            O::OffsetL2Factor => "OffsetL2Factor",
            O::Classes => "Classes",
            O::ClassWeights => "ClassWeights",
            O::ResponseNames => "ResponseNames",
        }
    }
}

impl DeepLearningLayerKind {
    pub fn option(self, name: &str) -> Option<DeepLearningLayerOption> {
        let option = DeepLearningLayerOption::from_name(name)?;
        self.supports_option(option).then_some(option)
    }

    pub const fn supports_option(self, option: DeepLearningLayerOption) -> bool {
        use DeepLearningLayerKind as K;
        use DeepLearningLayerOption as O;
        if matches!(option, O::Name | O::Description) {
            return true;
        }
        match self {
            K::FeatureInput | K::SequenceInput => input::supports(self, option),
            K::FullyConnected => dense::supports(option),
            K::Convolution1D => convolution::supports(option),
            K::Lstm | K::Bilstm => recurrent::supports(option),
            K::LayerNormalization => normalization::supports(option),
            K::ClassificationOutput | K::RegressionOutput => output::supports(self, option),
            K::GlobalAveragePooling1D | K::Relu | K::Elu | K::Softmax => false,
        }
    }
}
