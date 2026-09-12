use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

pub(in crate::builtins) const MAX_COMBVEC_COLUMNS: usize = 1_000_000;
pub(in crate::builtins) const MAX_PAD_ELEMENTS: usize = 10_000_000;
pub(in crate::builtins) const DLARRAY_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("dlarray");
pub(in crate::builtins) const LAYER_GRAPH_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.LayerGraph");
pub(in crate::builtins) const FULLY_CONNECTED_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.FullyConnectedLayer");
pub(in crate::builtins) const RELU_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.ReLULayer");
pub(in crate::builtins) const ELU_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.ELULayer");
pub(in crate::builtins) const SOFTMAX_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.SoftmaxLayer");
pub(in crate::builtins) const FEATURE_INPUT_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.FeatureInputLayer");
pub(in crate::builtins) const CLASSIFICATION_OUTPUT_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.ClassificationOutputLayer");
pub(in crate::builtins) const REGRESSION_OUTPUT_LAYER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.layer.RegressionOutputLayer");
pub(in crate::builtins) const TRAINING_OPTIONS_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.TrainingOptions");

pub(super) const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when:
        "Inputs or name-value options do not match the supported Deep Learning compatibility forms.",
    message: "deep learning builtin received invalid input",
};

pub(super) const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "The requested operation requires training, autodiff, export, or UI infrastructure outside this compatibility slice.",
    message: "deep learning operation is not supported in this slice",
};

pub(super) const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_UNSUPPORTED];

pub(super) static DLARRAY_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DLARRAY_CLASS);

pub(super) const OUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Deep Learning Toolbox compatibility object.",
}];

pub(super) const OUT_ARRAY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or cell array output.",
}];

pub(super) const IN_REST: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Builtin-specific positional and name-value arguments.",
}];

pub(super) const OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "obj = deepLearningBuiltin(args...)",
        inputs: &IN_REST,
        outputs: &OUT_OBJECT,
    }];

pub(super) const ARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = deepLearningUtility(args...)",
    inputs: &IN_REST,
    outputs: &OUT_ARRAY,
}];

pub(super) const OUT_ADAMUPDATE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "params",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Updated numeric parameters.",
    },
    BuiltinParamDescriptor {
        name: "averageGrad",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Updated first-moment moving average.",
    },
    BuiltinParamDescriptor {
        name: "averageSqGrad",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Updated second-moment moving average.",
    },
];

pub(super) const ADAMUPDATE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[params, averageGrad, averageSqGrad] = adamupdate(params, grad, averageGrad, averageSqGrad, iteration)",
        inputs: &IN_REST,
        outputs: &OUT_ADAMUPDATE,
    },
    BuiltinSignatureDescriptor {
        label: "[params, averageGrad, averageSqGrad] = adamupdate(params, grad, averageGrad, averageSqGrad, iteration, learnRate, gradDecay, sqGradDecay, epsilon)",
        inputs: &IN_REST,
        outputs: &OUT_ADAMUPDATE,
    },
];

pub(super) const OUT_VARARG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Outputs returned by the invoked function.",
}];

pub(super) const DLFEVAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle to evaluate.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments forwarded to the function handle.",
    },
];

pub(super) const DLFEVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "varargout = dlfeval(fun, args...)",
        inputs: &DLFEVAL_INPUTS,
        outputs: &OUT_VARARG,
    }];

pub(super) const DLUPDATE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle applied to matching leaves in the parameter trees.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more compatible parameter trees.",
    },
];

pub(super) const DLUPDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "varargout = dlupdate(fun, args...)",
        inputs: &DLUPDATE_INPUTS,
        outputs: &OUT_VARARG,
    }];

pub(super) const DLGRADIENT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "loss",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar traced dlarray loss.",
    },
    BuiltinParamDescriptor {
        name: "targets",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Traced dlarray values, learnables, or dlnetworks to differentiate.",
    },
];

pub(super) const DLGRADIENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "varargout = dlgradient(loss, targets...)",
        inputs: &DLGRADIENT_INPUTS,
        outputs: &OUT_VARARG,
    }];

pub const OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ADAMUPDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADAMUPDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DLFEVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLFEVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DLUPDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLUPDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DLGRADIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DLGRADIENT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
