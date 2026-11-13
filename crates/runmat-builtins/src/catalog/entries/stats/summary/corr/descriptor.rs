use crate::*;
const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise correlation coefficients.",
}];

const OUTPUT_R_P: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pairwise correlation coefficients.",
    },
    BuiltinParamDescriptor {
        name: "PValue",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "P-values for testing the hypothesis of no correlation.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations by variables.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Second observations-by-variables input.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Type, Rows, Tail, and Weights.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const SIGNATURES_WITH_P: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "R = corr(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_R_P,
    },
];

pub const CORR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.corr.INVALID_ARGUMENT",
    identifier: Some("RunMat:corr:InvalidArgument"),
    when: "Inputs, row counts, Type, Rows, Tail, or Weights options are malformed or unsupported.",
    message: "corr: invalid argument",
};

pub const CORR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.corr.INTERNAL",
    identifier: Some("RunMat:corr:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "corr: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [CORR_ERROR_INVALID_ARGUMENT, CORR_ERROR_INTERNAL];

pub const CORR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES_WITH_P,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
