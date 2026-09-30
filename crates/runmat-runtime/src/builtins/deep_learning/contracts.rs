use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

pub(in crate::builtins) const MAX_COMBVEC_COLUMNS: usize = 1_000_000;
pub(in crate::builtins) const MAX_PAD_ELEMENTS: usize = 10_000_000;
pub(in crate::builtins) const DLARRAY_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DL_ARRAY;
pub(in crate::builtins) const LAYER_GRAPH_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("nnet.cnn.LayerGraph");
pub(in crate::builtins) use runmat_builtins::{
    ELU_LAYER_CLASS, FEATURE_INPUT_LAYER_CLASS, FULLY_CONNECTED_LAYER_CLASS, RELU_LAYER_CLASS,
    SOFTMAX_LAYER_CLASS,
};

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
    when: "The requested form, precision, placement, or provider capability is not supported for this operation.",
    message: "deep learning operation is not supported for these inputs",
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
