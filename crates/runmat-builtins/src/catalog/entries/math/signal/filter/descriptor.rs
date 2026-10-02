use crate::*;

const FILTER_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered signal output.",
}];

const FILTER_OUTPUT_Y_ZF: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filtered signal output.",
    },
    BuiltinParamDescriptor {
        name: "zf",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Final filter state for warm-starting a subsequent call.",
    },
];

const FILTER_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Signal values to filter.",
    },
];

const FILTER_INPUTS_WITH_ZI: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Signal values to filter.",
    },
    BuiltinParamDescriptor {
        name: "zi",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Initial filter state (use [] for default zero state).",
    },
];

const FILTER_INPUTS_WITH_ZI_DIM: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Signal values to filter.",
    },
    BuiltinParamDescriptor {
        name: "zi",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Initial filter state (use [] for default zero state).",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("first non-singleton dimension"),
        description: "Dimension to operate along.",
    },
];

const FILTER_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "y = filter(b, a, x)",
        inputs: &FILTER_INPUTS_CORE,
        outputs: &FILTER_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = filter(b, a, x, zi)",
        inputs: &FILTER_INPUTS_WITH_ZI,
        outputs: &FILTER_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = filter(b, a, x, zi, dim)",
        inputs: &FILTER_INPUTS_WITH_ZI_DIM,
        outputs: &FILTER_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y, zf] = filter(b, a, x)",
        inputs: &FILTER_INPUTS_CORE,
        outputs: &FILTER_OUTPUT_Y_ZF,
    },
    BuiltinSignatureDescriptor {
        label: "[y, zf] = filter(b, a, x, zi)",
        inputs: &FILTER_INPUTS_WITH_ZI,
        outputs: &FILTER_OUTPUT_Y_ZF,
    },
    BuiltinSignatureDescriptor {
        label: "[y, zf] = filter(b, a, x, zi, dim)",
        inputs: &FILTER_INPUTS_WITH_ZI_DIM,
        outputs: &FILTER_OUTPUT_Y_ZF,
    },
];

pub const FILTER_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.ARG_COUNT",
    identifier: Some("RunMat:filter:ArgCount"),
    when: "More than five input arguments are provided.",
    message: "filter: expected between three and five input arguments",
};

pub const FILTER_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.INVALID_DIMENSION",
    identifier: Some("RunMat:filter:InvalidDimension"),
    when: "dim is missing, non-numeric, non-integer, or less than one.",
    message: "filter: dimension must be numeric and >= 1",
};

pub const FILTER_ERROR_EMPTY_DENOMINATOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.EMPTY_DENOMINATOR",
    identifier: Some("RunMat:filter:EmptyDenominator"),
    when: "Denominator coefficient vector is empty.",
    message: "filter: denominator coefficients cannot be empty",
};

pub const FILTER_ERROR_EMPTY_NUMERATOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.EMPTY_NUMERATOR",
    identifier: Some("RunMat:filter:EmptyNumerator"),
    when: "Numerator coefficient vector is empty.",
    message: "filter: numerator coefficients cannot be empty",
};

pub const FILTER_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:filter:InvalidCoefficients"),
    when: "Coefficient inputs are non-numeric or non-vector values.",
    message: "filter: invalid coefficient input",
};

pub const FILTER_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.INVALID_SIGNAL",
    identifier: Some("RunMat:filter:InvalidSignal"),
    when: "Signal input is non-numeric/logical.",
    message: "filter: invalid signal input",
};

pub const FILTER_ERROR_INVALID_INITIAL_STATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.INVALID_INITIAL_STATE",
    identifier: Some("RunMat:filter:InvalidInitialState"),
    when: "Initial state input is invalid for current filter/order configuration.",
    message: "filter: invalid initial state input",
};

pub const FILTER_ERROR_DENOMINATOR_A1_ZERO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.DENOMINATOR_A1_ZERO",
    identifier: Some("RunMat:filter:DenominatorLeadingZero"),
    when: "The leading denominator coefficient a(1) is zero.",
    message: "filter: denominator coefficient a(1) must be non-zero",
};

pub const FILTER_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.GATHER_FAILED",
    identifier: Some("RunMat:filter:GatherFailed"),
    when: "GPU values fail to gather for host normalization/fallback.",
    message: "filter: failed to gather GPU value",
};

pub const FILTER_ERROR_PROVIDER_UPLOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.PROVIDER_UPLOAD",
    identifier: Some("RunMat:filter:ProviderUploadFailed"),
    when: "Provider-side upload for coefficients or initial state fails.",
    message: "filter: failed to upload provider input",
};

pub const FILTER_ERROR_PROVIDER_EXECUTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.PROVIDER_EXECUTION",
    identifier: Some("RunMat:filter:ProviderFailed"),
    when: "The provider iir_filter hook fails with an operational error.",
    message: "filter: GPU provider iir_filter failed",
};

pub const FILTER_ERROR_PROVIDER_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.PROVIDER_OUTPUT",
    identifier: Some("RunMat:filter:InvalidProviderOutput"),
    when: "The provider returns missing, malformed, foreign, stale, or aliased filter results.",
    message: "filter: invalid provider filtering output",
};

pub const FILTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTER.INTERNAL",
    identifier: Some("RunMat:filter:Internal"),
    when: "Internal tensor/state/index processing fails.",
    message: "filter: internal error",
};

const FILTER_ERRORS: [BuiltinErrorDescriptor; 13] = [
    FILTER_ERROR_ARG_COUNT,
    FILTER_ERROR_INVALID_DIMENSION,
    FILTER_ERROR_EMPTY_DENOMINATOR,
    FILTER_ERROR_EMPTY_NUMERATOR,
    FILTER_ERROR_INVALID_COEFFICIENTS,
    FILTER_ERROR_INVALID_SIGNAL,
    FILTER_ERROR_INVALID_INITIAL_STATE,
    FILTER_ERROR_DENOMINATOR_A1_ZERO,
    FILTER_ERROR_GATHER_FAILED,
    FILTER_ERROR_PROVIDER_UPLOAD,
    FILTER_ERROR_PROVIDER_EXECUTION,
    FILTER_ERROR_PROVIDER_OUTPUT,
    FILTER_ERROR_INTERNAL,
];

pub const FILTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILTER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILTER_ERRORS,
};
