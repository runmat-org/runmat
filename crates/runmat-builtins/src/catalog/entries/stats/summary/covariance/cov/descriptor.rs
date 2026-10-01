use crate::*;
const COV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Covariance matrix.",
}];

const COV_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations (rows are observations, columns are variables).",
}];

const COV_INPUTS_X_Y_OR_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y_or_w",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset (Y) or weight vector (w), depending on shape/position.",
    },
];

const COV_INPUTS_X_NORMALIZATION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "normalization",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0"),
        description: "Normalization flag: 0 (unbiased) or 1 (biased).",
    },
];

const COV_INPUTS_X_ROWS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "rows_option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"all\""),
        description: "Rows handling mode: 'all', 'omitrows', or 'partialrows'.",
    },
];

const COV_INPUTS_X_Y_OPT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset with matching size (or equal vector length).",
    },
    BuiltinParamDescriptor {
        name: "opt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalization flag or rows option.",
    },
];

const COV_INPUTS_X_Y_W: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset with matching size (or equal vector length).",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Weight vector with one weight per observation row.",
    },
];

const COV_INPUTS_X_Y_W_OPT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset with matching size (or equal vector length).",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Weight vector with one weight per observation row.",
    },
    BuiltinParamDescriptor {
        name: "opt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalization flag or rows option.",
    },
];

const COV_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "C = cov(X)",
        inputs: &COV_INPUTS_X,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y_or_w)",
        inputs: &COV_INPUTS_X_Y_OR_W,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, normalization)",
        inputs: &COV_INPUTS_X_NORMALIZATION,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, rows_option)",
        inputs: &COV_INPUTS_X_ROWS,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y, opt)",
        inputs: &COV_INPUTS_X_Y_OPT,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y, w)",
        inputs: &COV_INPUTS_X_Y_W,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y, w, opt)",
        inputs: &COV_INPUTS_X_Y_W_OPT,
        outputs: &COV_OUTPUT,
    },
];

pub const COV_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.INVALID_ARGUMENT",
    identifier: Some("RunMat:cov:InvalidArgument"),
    when: "Arguments are malformed or unsupported for cov.",
    message: "cov: invalid argument",
};

pub const COV_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:cov:ComplexUnsupported"),
    when: "Any argument is complex-valued.",
    message: "cov: complex inputs are not supported yet",
};

pub const COV_ERROR_ROWS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.ROWS_MISMATCH",
    identifier: Some("RunMat:cov:RowsMismatch"),
    when: "Two input datasets do not have the same size or equal vector lengths.",
    message: "cov: paired inputs must have the same size",
};

pub const COV_ERROR_NORMALIZATION_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.NORMALIZATION_INVALID",
    identifier: Some("RunMat:cov:NormalizationInvalid"),
    when: "Normalization flag is non-finite, non-integer, or not 0/1.",
    message: "cov: normalization flag is invalid",
};

pub const COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.COV.WEIGHT_VECTOR_LENGTH_MISMATCH",
        identifier: Some("RunMat:cov:WeightVectorLengthMismatch"),
        when: "Weight vector length does not match observation row count.",
        message: "cov: weight vector length mismatch",
    };

pub const COV_ERROR_ROWS_OPTION_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.ROWS_OPTION_UNKNOWN",
    identifier: Some("RunMat:cov:RowsOptionUnknown"),
    when: "Rows option is not one of all/omitrows/partialrows.",
    message: "cov: unknown rows option",
};

pub const COV_ERROR_NORMALIZATION_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.NORMALIZATION_DUPLICATE",
    identifier: Some("RunMat:cov:NormalizationDuplicate"),
    when: "Normalization flag is provided more than once.",
    message: "cov: normalization flag specified more than once",
};

pub const COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.TOO_MANY_ARRAY_ARGUMENTS",
    identifier: Some("RunMat:cov:TooManyArrayArguments"),
    when: "More than two data arrays (or Y plus weight) are provided.",
    message: "cov: too many array arguments",
};

pub const COV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.INTERNAL",
    identifier: Some("RunMat:cov:Internal"),
    when: "Internal tensor conversion/allocation or covariance computation fails.",
    message: "cov: internal operation failed",
};

pub const COV_ERRORS: [BuiltinErrorDescriptor; 9] = [
    COV_ERROR_INVALID_ARGUMENT,
    COV_ERROR_COMPLEX_UNSUPPORTED,
    COV_ERROR_ROWS_MISMATCH,
    COV_ERROR_NORMALIZATION_INVALID,
    COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
    COV_ERROR_ROWS_OPTION_UNKNOWN,
    COV_ERROR_NORMALIZATION_DUPLICATE,
    COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS,
    COV_ERROR_INTERNAL,
];

pub const COV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COV_ERRORS,
};
