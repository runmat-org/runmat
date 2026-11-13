use super::super::sample_rate_contract::{
    SAMPLE_ERROR_ARG_COUNT, SAMPLE_ERROR_BUILD_OUTPUT, SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_INVALID_FACTOR, SAMPLE_ERROR_INVALID_INPUT, SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_SIZE_OVERFLOW,
};
use crate::*;
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input signal array.",
};
const P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer upsampling factor.",
};
const Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer downsampling factor.",
};
const N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("10"),
    description: "Half-length filter order factor.",
};
const BETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BETA",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("5"),
    description: "Kaiser window shape parameter.",
};
const B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "FIR antialiasing coefficients.",
};
const DIMENSION_KEY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Dimension",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Name-value key selecting the operated dimension.",
};
const DIMENSION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "DIM",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive one-based operating dimension.",
};
const CORE: [BuiltinParamDescriptor; 3] = [X, P, Q];
const FILTER: [BuiltinParamDescriptor; 4] = [X, P, Q, B];
const DESIGN: [BuiltinParamDescriptor; 5] = [X, P, Q, N, BETA];
const DESIGN_N: [BuiltinParamDescriptor; 4] = [X, P, Q, N];
const DIMENSION: [BuiltinParamDescriptor; 5] = [X, P, Q, DIMENSION_KEY, DIMENSION_VALUE];
const FILTER_DIMENSION: [BuiltinParamDescriptor; 6] = [X, P, Q, B, DIMENSION_KEY, DIMENSION_VALUE];
const VARIADIC_FILTER: [BuiltinParamDescriptor; 4] = [
    X,
    P,
    Q,
    BuiltinParamDescriptor {
        arity: BuiltinParamArity::Variadic,
        ..B
    },
];
const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Resampled signal.",
};
const USED_FILTER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "FIR coefficients used for resampling.",
};
const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [Y];
const OUTPUT_Y_B: [BuiltinParamDescriptor; 2] = [Y, USED_FILTER];
const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q)",
        inputs: &CORE,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, N)",
        inputs: &DESIGN_N,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, N, BETA)",
        inputs: &DESIGN,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, B)",
        inputs: &FILTER,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, 'Dimension', DIM)",
        inputs: &DIMENSION,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, B, 'Dimension', DIM)",
        inputs: &FILTER_DIMENSION,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, ___)",
        inputs: &DESIGN,
        outputs: &OUTPUT_Y_B,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, B, ___)",
        inputs: &VARIADIC_FILTER,
        outputs: &OUTPUT_Y_B,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, 'Dimension', DIM)",
        inputs: &DIMENSION,
        outputs: &OUTPUT_Y_B,
    },
];
pub const SAMPLE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_OPTION",
    identifier: Some("RunMat:sampleRate:InvalidOption"),
    when: "A conversion option, filter vector, or dimension selector is invalid.",
    message: "sample-rate builtin: invalid option",
};
pub const RESAMPLE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.OUTPUT_COUNT",
    identifier: Some("RunMat:sampleRate:OutputCount"),
    when: "More than two outputs are requested.",
    message: "sample-rate builtin: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 9] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
    SAMPLE_ERROR_INVALID_OPTION,
    RESAMPLE_ERROR_OUTPUT_COUNT,
];
pub const RESAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
