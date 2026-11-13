use super::errors::HIST_ERRORS;
use super::signatures::HIST_SIGNATURES;
use crate::*;

pub const HIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HIST_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HIST_ERRORS,
};
