use crate::*;
const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation coefficient matrix.",
};
const OUTPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "P-values for testing zero correlation.",
};
const OUTPUT_RL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "RL",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Lower confidence bounds for the correlation coefficients.",
};
const OUTPUT_RU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "RU",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Upper confidence bounds for the correlation coefficients.",
};
const OUTPUTS_R: [BuiltinParamDescriptor; 1] = [OUTPUT_R];
const OUTPUTS_R_P: [BuiltinParamDescriptor; 2] = [OUTPUT_R, OUTPUT_P];
const OUTPUTS_FULL: [BuiltinParamDescriptor; 4] = [OUTPUT_R, OUTPUT_P, OUTPUT_RL, OUTPUT_RU];

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations (rows are observations, columns are variables).",
};
const PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second array with the same size as A; both arrays are vectorized.",
};
const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha and Rows name-value arguments.",
};
const INPUTS_A: [BuiltinParamDescriptor; 1] = [PARAM_A];
const INPUTS_A_B: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_B];
const INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_OPTIONS];
const INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_B, PARAM_OPTIONS];

const CORRCOEF_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "R = corrcoef(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corrcoef(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P] = corrcoef(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P] = corrcoef(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUTS_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P, RL, RU] = corrcoef(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_FULL,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P, RL, RU] = corrcoef(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUTS_FULL,
    },
    BuiltinSignatureDescriptor {
        label: "___ = corrcoef(A, Name, Value)",
        inputs: &INPUTS_A_OPTIONS,
        outputs: &OUTPUTS_FULL,
    },
    BuiltinSignatureDescriptor {
        label: "___ = corrcoef(A, B, Name, Value)",
        inputs: &INPUTS_A_B_OPTIONS,
        outputs: &OUTPUTS_FULL,
    },
];

pub const CORRCOEF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.INVALID_ARGUMENT",
    identifier: Some("RunMat:corrcoef:InvalidArgument"),
    when: "Arguments are malformed or unsupported for corrcoef.",
    message: "corrcoef: invalid argument",
};

pub const CORRCOEF_ERROR_COMPLEX_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.COMPLEX_OUTPUTS",
    identifier: Some("RunMat:corrcoef:ComplexOutputs"),
    when: "P-values or confidence bounds are requested for complex correlation coefficients.",
    message: "corrcoef: P-values and confidence bounds are invalid for complex coefficients",
};

pub const CORRCOEF_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.SIZE_MISMATCH",
    identifier: Some("RunMat:corrcoef:SizeMismatch"),
    when: "A and B do not have the same size.",
    message: "corrcoef: A and B must have the same size",
};

pub const CORRCOEF_ERROR_ALPHA_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.ALPHA_INVALID",
    identifier: Some("RunMat:corrcoef:AlphaInvalid"),
    when: "Alpha is not a finite single/double scalar strictly between zero and one.",
    message: "corrcoef: Alpha must be a floating scalar in the open interval (0,1)",
};

pub const CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.ROWS_OPTION_UNKNOWN",
    identifier: Some("RunMat:corrcoef:RowsOptionUnknown"),
    when: "Rows option value is not supported.",
    message: "corrcoef: unknown rows option",
};

pub const CORRCOEF_ERROR_OPTION_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.OPTION_DUPLICATE",
    identifier: Some("RunMat:corrcoef:OptionDuplicate"),
    when: "Alpha or Rows is specified more than once.",
    message: "corrcoef: option specified more than once",
};

pub const CORRCOEF_ERROR_ROWS_OPTION_MALFORMED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.ROWS_OPTION_MALFORMED",
    identifier: Some("RunMat:corrcoef:RowsOptionMalformed"),
    when: "Rows keyword is not followed by a valid string option.",
    message: "corrcoef: rows option is malformed",
};

pub const CORRCOEF_ERROR_OPTION_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.OPTION_UNKNOWN",
    identifier: Some("RunMat:corrcoef:OptionUnknown"),
    when: "An unknown option keyword is provided.",
    message: "corrcoef: unknown option",
};

pub const CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.TOO_MANY_INPUT_ARRAYS",
    identifier: Some("RunMat:corrcoef:TooManyInputArrays"),
    when: "More than two data arrays are provided.",
    message: "corrcoef: too many input arrays",
};

pub const CORRCOEF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.INTERNAL",
    identifier: Some("RunMat:corrcoef:Internal"),
    when: "Internal tensor conversion/allocation or corrcoef computation fails.",
    message: "corrcoef: internal operation failed",
};

pub const CORRCOEF_ERRORS: [BuiltinErrorDescriptor; 10] = [
    CORRCOEF_ERROR_INVALID_ARGUMENT,
    CORRCOEF_ERROR_COMPLEX_OUTPUTS,
    CORRCOEF_ERROR_SIZE_MISMATCH,
    CORRCOEF_ERROR_ALPHA_INVALID,
    CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN,
    CORRCOEF_ERROR_OPTION_DUPLICATE,
    CORRCOEF_ERROR_ROWS_OPTION_MALFORMED,
    CORRCOEF_ERROR_OPTION_UNKNOWN,
    CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS,
    CORRCOEF_ERROR_INTERNAL,
];

pub const CORRCOEF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CORRCOEF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CORRCOEF_ERRORS,
};
