//! Domain-owned sample-rate signature fields and shared error identity.

use crate::{BuiltinErrorDescriptor, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType};

pub(super) const SAMPLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample-rate-adjusted signal.",
}];
pub(super) const SAMPLE_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive integer sample-rate factor.",
    },
];
pub(super) const SAMPLE_INPUTS_PHASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive integer sample-rate factor.",
    },
    BuiltinParamDescriptor {
        name: "PHASE",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Integer offset satisfying 0 <= PHASE < N.",
    },
];

pub const SAMPLE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.ARG_COUNT",
    identifier: Some("RunMat:sampleRate:ArgCount"),
    when: "Required arguments are missing or more than three inputs are provided.",
    message: "sample-rate builtin: expected X, N, and optional PHASE",
};
pub const SAMPLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_INPUT",
    identifier: Some("RunMat:sampleRate:InvalidInput"),
    when: "X is not numeric, logical, complex, or a gpuArray containing numeric values.",
    message: "sample-rate builtin: X must be numeric or logical",
};
pub const SAMPLE_ERROR_INVALID_FACTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_FACTOR",
    identifier: Some("RunMat:sampleRate:InvalidFactor"),
    when: "N is not a finite positive integer scalar.",
    message: "sample-rate builtin: N must be a positive integer scalar",
};
pub const SAMPLE_ERROR_INVALID_PHASE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_PHASE",
    identifier: Some("RunMat:sampleRate:InvalidPhase"),
    when: "PHASE is not an integer scalar in the range 0 <= PHASE < N.",
    message: "sample-rate builtin: PHASE must be an integer scalar with 0 <= PHASE < N",
};
pub const SAMPLE_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.GATHER_FAILED",
    identifier: Some("RunMat:sampleRate:GatherFailed"),
    when: "A gpuArray input cannot be gathered for host sample-rate conversion.",
    message: "sample-rate builtin: failed to gather GPU input",
};
pub const SAMPLE_ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.SIZE_OVERFLOW",
    identifier: Some("RunMat:sampleRate:SizeOverflow"),
    when: "The requested output shape overflows host allocation limits.",
    message: "sample-rate builtin: requested output is too large",
};
pub const SAMPLE_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.BUILD_OUTPUT",
    identifier: Some("RunMat:sampleRate:BuildOutput"),
    when: "Output tensor construction fails.",
    message: "sample-rate builtin: failed to build output",
};
pub(super) const BASIC_SAMPLE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
];
