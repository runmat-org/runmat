use crate::*;

pub const TRAINING_OPTIONS_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.TrainingOptions");

pub fn canonical_training_option_name(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "maxepochs" => "MaxEpochs",
        "minibatchsize" => "MiniBatchSize",
        "initiallearnrate" => "InitialLearnRate",
        "shuffle" => "Shuffle",
        "verbose" => "Verbose",
        "plots" => "Plots",
        "executionenvironment" => "ExecutionEnvironment",
        "validationdata" => "ValidationData",
        "validationfrequency" => "ValidationFrequency",
        "learnrateschedule" => "LearnRateSchedule",
        "learnratedropperiod" => "LearnRateDropPeriod",
        "learnratedropfactor" => "LearnRateDropFactor",
        "gradientthreshold" => "GradientThreshold",
        "l2regularization" => "L2Regularization",
        "momentum" => "Momentum",
        "squaredgradientdecayfactor" => "SquaredGradientDecayFactor",
        "gradientdecayfactor" => "GradientDecayFactor",
        "epsilon" => "Epsilon",
        other => return other.to_owned(),
    }
    .to_owned()
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
    name: "MaxEpochs, MiniBatchSize, and other numeric training options",
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
