use super::super::sample_rate_contract::{
    BASIC_SAMPLE_ERRORS, SAMPLE_INPUTS_CORE, SAMPLE_INPUTS_PHASE, SAMPLE_OUTPUT,
};
use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinSignatureDescriptor,
};
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = upsample(X, N)",
        inputs: &SAMPLE_INPUTS_CORE,
        outputs: &SAMPLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = upsample(X, N, PHASE)",
        inputs: &SAMPLE_INPUTS_PHASE,
        outputs: &SAMPLE_OUTPUT,
    },
];
pub const UPSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BASIC_SAMPLE_ERRORS,
};
pub use super::super::sample_rate_contract::{
    SAMPLE_ERROR_ARG_COUNT as UPSAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_BUILD_OUTPUT as UPSAMPLE_ERROR_BUILD_OUTPUT,
    SAMPLE_ERROR_GATHER_FAILED as UPSAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_INVALID_FACTOR as UPSAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_INPUT as UPSAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_PHASE as UPSAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_SIZE_OVERFLOW as UPSAMPLE_ERROR_SIZE_OVERFLOW,
};
