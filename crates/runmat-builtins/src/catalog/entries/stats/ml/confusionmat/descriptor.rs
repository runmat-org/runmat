use crate::*;
const OUTPUT_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Confusion matrix with true labels in rows and predicted labels in columns.",
};

const OUTPUT_ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "order",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Class labels corresponding to rows and columns of C.",
};

const PARAM_GROUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "group",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True class labels.",
};

const PARAM_GROUPHAT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "grouphat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted class labels.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Order.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 2] = [PARAM_GROUP, PARAM_GROUPHAT];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_GROUP, PARAM_GROUPHAT, PARAM_OPTIONS];
const OUTPUTS_C: [BuiltinParamDescriptor; 1] = [OUTPUT_C];
const OUTPUTS_ALL: [BuiltinParamDescriptor; 2] = [OUTPUT_C, OUTPUT_ORDER];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "C = confusionmat(group, grouphat)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUTS_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = confusionmat(group, grouphat, Name, Value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C,order] = confusionmat(___)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_ALL,
    },
];

pub const CONFUSIONMAT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONFUSIONMAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:confusionmat:InvalidArgument"),
    when: "Label vectors, dimensions, order labels, or name-value options are malformed.",
    message: "confusionmat: invalid argument",
};

pub const CONFUSIONMAT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONFUSIONMAT.INTERNAL",
    identifier: Some("RunMat:confusionmat:Internal"),
    when: "RunMat cannot allocate or construct confusion matrix outputs.",
    message: "confusionmat: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    CONFUSIONMAT_ERROR_INVALID_ARGUMENT,
    CONFUSIONMAT_ERROR_INTERNAL,
];

pub const CONFUSIONMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
