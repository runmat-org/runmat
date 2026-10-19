use super::super::sample_rate_contract::{
    BASIC_SAMPLE_ERRORS, SAMPLE_INPUTS_CORE, SAMPLE_INPUTS_PHASE, SAMPLE_OUTPUT,
};
pub use super::super::sample_rate_contract::{
    SAMPLE_ERROR_ARG_COUNT, SAMPLE_ERROR_BUILD_OUTPUT, SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_INVALID_FACTOR, SAMPLE_ERROR_INVALID_INPUT, SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_SIZE_OVERFLOW,
};
use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinSignatureDescriptor,
};

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = downsample(X, N)",
        inputs: &SAMPLE_INPUTS_CORE,
        outputs: &SAMPLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = downsample(X, N, PHASE)",
        inputs: &SAMPLE_INPUTS_PHASE,
        outputs: &SAMPLE_OUTPUT,
    },
];
pub const DOWNSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BASIC_SAMPLE_ERRORS,
};
