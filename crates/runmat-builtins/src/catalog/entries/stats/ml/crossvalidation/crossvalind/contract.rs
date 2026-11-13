use crate::*;

const PARAM_METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cross-validation method: KFold, HoldOut, LeaveMOut, or Resubstitution.",
};

const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation count.",
};

const PARAM_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Fold count, holdout fraction/count, or leave-M-out count.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Classes and Min.",
};

const OUTPUT_INDICES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "indices",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fold indices for KFold, or a logical training mask for partition methods.",
}];

const OUTPUT_TRAIN_TEST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "train",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical training-set mask.",
    },
    BuiltinParamDescriptor {
        name: "test",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Logical test-set mask.",
    },
];

const INPUTS_METHOD_N_VALUE: [BuiltinParamDescriptor; 3] = [PARAM_METHOD, PARAM_N, PARAM_VALUE];
const INPUTS_METHOD_N_VALUE_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_METHOD, PARAM_N, PARAM_VALUE, PARAM_OPTIONS];
const INPUTS_METHOD_N: [BuiltinParamDescriptor; 2] = [PARAM_METHOD, PARAM_N];

const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "indices = crossvalind('KFold', n, k)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_INDICES,
    },
    BuiltinSignatureDescriptor {
        label: "indices = crossvalind('KFold', n, k, 'Classes', group)",
        inputs: &INPUTS_METHOD_N_VALUE_OPTIONS,
        outputs: &OUTPUT_INDICES,
    },
    BuiltinSignatureDescriptor {
        label: "train = crossvalind('HoldOut', n, p)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_INDICES,
    },
    BuiltinSignatureDescriptor {
        label: "[train, test] = crossvalind('HoldOut', n, p)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_TRAIN_TEST,
    },
    BuiltinSignatureDescriptor {
        label: "[train, test] = crossvalind('LeaveMOut', n, m)",
        inputs: &INPUTS_METHOD_N_VALUE,
        outputs: &OUTPUT_TRAIN_TEST,
    },
    BuiltinSignatureDescriptor {
        label: "[train, test] = crossvalind('Resubstitution', n)",
        inputs: &INPUTS_METHOD_N,
        outputs: &OUTPUT_TRAIN_TEST,
    },
];

pub const CROSSVALIND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSSVALIND.INVALID_ARGUMENT",
    identifier: Some("RunMat:crossvalind:InvalidArgument"),
    when: "Inputs, method names, partition values, options, or output counts are malformed.",
    message: "crossvalind: invalid argument",
};

pub const CROSSVALIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSSVALIND.INTERNAL",
    identifier: Some("RunMat:crossvalind:Internal"),
    when: "RunMat cannot allocate or construct the requested partition output.",
    message: "crossvalind: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    CROSSVALIND_ERROR_INVALID_ARGUMENT,
    CROSSVALIND_ERROR_INTERNAL,
];

pub const CROSSVALIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
