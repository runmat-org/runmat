use crate::*;

const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real, finite, two-dimensional in-memory predictors with at least one observation in rows. A resident array is gathered only after host controls are admitted.",
};
const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Regression matrix with one row per observation, or classification labels (numeric, string, categorical) or one-hot matrix.",
};
const LAYERS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "layers",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Supported sequential layer collection or simple layerGraph object; branching execution is not supported.",
};
const NET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "net",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "dlnetwork, trained network, simple layerGraph, or supported sequential layer collection.",
};
const LOSS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lossFcn",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Built-in mse/mean-squared-error regression or crossentropy classification loss; function handles reject.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "trainingOptions object with supported sgdm or adam host/auto training controls.",
};
const NETWORK_INPUTS: [BuiltinParamDescriptor; 4] = [X, Y, LAYERS, OPTIONS];
const TRAINNET_INPUTS: [BuiltinParamDescriptor; 5] = [X, Y, NET, LOSS, OPTIONS];
const TRAINED_NETWORK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "net",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Trained SeriesNetwork or DAGNetwork compatibility object with learned weights and training metadata.",
}];
const TRAINED_DLNETWORK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "net",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Trained dlnetwork compatibility object with learned weights and training metadata.",
}];
const INFO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Training information structure containing solver, epochs, iterations, loss history and stop reason.",
};
const NETWORK_WITH_INFO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "net", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
        default: None,
        description: "Trained SeriesNetwork or DAGNetwork compatibility object with learned weights and training metadata.",
    },
    INFO,
];
const DLNETWORK_WITH_INFO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "net", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
        default: None,
        description: "Trained dlnetwork compatibility object with learned weights and training metadata.",
    },
    BuiltinParamDescriptor {
        name: "info", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
        default: None,
        description: "Training information structure containing solver, epochs, iterations, loss history and stop reason.",
    },
];
const NETWORK_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "net = trainNetwork(X, Y, layers, options)",
        inputs: &NETWORK_INPUTS,
        outputs: &TRAINED_NETWORK,
    },
    BuiltinSignatureDescriptor {
        label: "[net, info] = trainNetwork(X, Y, layers, options)",
        inputs: &NETWORK_INPUTS,
        outputs: &NETWORK_WITH_INFO,
    },
];
const TRAINNET_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "net = trainnet(X, Y, net, lossFcn, options)",
        inputs: &TRAINNET_INPUTS,
        outputs: &TRAINED_DLNETWORK,
    },
    BuiltinSignatureDescriptor {
        label: "[net, info] = trainnet(X, Y, net, lossFcn, options)",
        inputs: &TRAINNET_INPUTS,
        outputs: &DLNETWORK_WITH_INFO,
    },
];

pub const SUPERVISED_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "Argument/output count, host or resident matrix metadata, targets, layer topology or parameters, loss name, consumed training option, or numeric training result is invalid.",
    message: "supervised training: invalid data, network, loss, option, or requested result",
};
const SUPERVISED_CANCELLED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.CANCELLED",
    identifier: Some("RunMat:Cancelled"),
    when: "The active execution is cancelled before or during a training epoch.",
    message: "supervised training: execution cancelled",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [SUPERVISED_INVALID_INPUT, SUPERVISED_CANCELLED];
pub const TRAIN_NETWORK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NETWORK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const TRAINNET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRAINNET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const STRUCTURAL_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "MaxEpochs, MiniBatchSize and other consumed integer-valued trainingOptions controls",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Controls are decoded from native integer storage and range-checked before training, without a binary64 detour.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "trainNetwork/trainnet(..., trainingOptions with typed integer controls)",
    inputs: &STRUCTURAL_CONTROLS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Predictor/target integer arrays enter floating training arithmetic; this control record does not assert per-class MATLAB predictor/target parity.",
}];
