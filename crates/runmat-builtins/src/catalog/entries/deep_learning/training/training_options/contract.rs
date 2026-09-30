use crate::*;

pub const TRAINING_OPTIONS_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.TrainingOptions");

/// Canonical training-configuration property identity. Parsing is confined to
/// the public name-value/object boundary; consumers match these roles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TrainingOptionKey {
    SolverName,
    MaxEpochs,
    MiniBatchSize,
    InitialLearnRate,
    Shuffle,
    Verbose,
    Plots,
    ExecutionEnvironment,
    ValidationData,
    ValidationFrequency,
    LearnRateSchedule,
    LearnRateDropPeriod,
    LearnRateDropFactor,
    GradientThreshold,
    L2Regularization,
    Momentum,
    SquaredGradientDecayFactor,
    GradientDecayFactor,
    Epsilon,
}

impl TrainingOptionKey {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "solvername" => Self::SolverName,
            "maxepochs" => Self::MaxEpochs,
            "minibatchsize" => Self::MiniBatchSize,
            "initiallearnrate" => Self::InitialLearnRate,
            "shuffle" => Self::Shuffle,
            "verbose" => Self::Verbose,
            "plots" => Self::Plots,
            "executionenvironment" => Self::ExecutionEnvironment,
            "validationdata" => Self::ValidationData,
            "validationfrequency" => Self::ValidationFrequency,
            "learnrateschedule" => Self::LearnRateSchedule,
            "learnratedropperiod" => Self::LearnRateDropPeriod,
            "learnratedropfactor" => Self::LearnRateDropFactor,
            "gradientthreshold" => Self::GradientThreshold,
            "l2regularization" => Self::L2Regularization,
            "momentum" => Self::Momentum,
            "squaredgradientdecayfactor" => Self::SquaredGradientDecayFactor,
            "gradientdecayfactor" => Self::GradientDecayFactor,
            "epsilon" => Self::Epsilon,
            _ => return None,
        })
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::SolverName => "SolverName",
            Self::MaxEpochs => "MaxEpochs",
            Self::MiniBatchSize => "MiniBatchSize",
            Self::InitialLearnRate => "InitialLearnRate",
            Self::Shuffle => "Shuffle",
            Self::Verbose => "Verbose",
            Self::Plots => "Plots",
            Self::ExecutionEnvironment => "ExecutionEnvironment",
            Self::ValidationData => "ValidationData",
            Self::ValidationFrequency => "ValidationFrequency",
            Self::LearnRateSchedule => "LearnRateSchedule",
            Self::LearnRateDropPeriod => "LearnRateDropPeriod",
            Self::LearnRateDropFactor => "LearnRateDropFactor",
            Self::GradientThreshold => "GradientThreshold",
            Self::L2Regularization => "L2Regularization",
            Self::Momentum => "Momentum",
            Self::SquaredGradientDecayFactor => "SquaredGradientDecayFactor",
            Self::GradientDecayFactor => "GradientDecayFactor",
            Self::Epsilon => "Epsilon",
        }
    }
}

pub fn canonical_training_option_name(name: &str) -> String {
    TrainingOptionKey::from_name(name)
        .map(|key| key.canonical_name().to_owned())
        .unwrap_or_else(|| name.to_ascii_lowercase())
}

const SOLVER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "solver",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host text naming the sgdm, adam, or rmsprop solver.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Paired training-option names and values; values are stored without provider download.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [SOLVER, OPTIONS];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host training-options metadata object.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "options = trainingOptions(solver, Name, Value, ...)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const TRAINING_OPTIONS_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The solver is unsupported, a name-value list is malformed, or more than one output is requested.",
    message: "trainingOptions: invalid solver, option list, or requested output count",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [TRAINING_OPTIONS_INVALID_INPUT];
pub const TRAINING_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "MaxEpochs and MiniBatchSize",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer option values remain exact in the metadata object; consuming training operations validate and decode supported controls separately.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "trainingOptions(solver, numericOption, integer_value)",
    inputs: &INTEGER_CONTROLS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Construction stores option values as supplied; it does not convert wide integer controls through double. Downstream training has its own supported-option and range checks.",
}];
