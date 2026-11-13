use super::errors::ERRORS;
use super::signatures::SIGNATURES;
use crate::*;

pub const HISTOGRAM2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
