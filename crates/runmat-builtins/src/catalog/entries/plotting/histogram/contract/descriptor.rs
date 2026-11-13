use super::errors::HISTOGRAM_ERRORS;
use super::signatures::HISTOGRAM_SIGNATURES;
use crate::*;

pub const HISTOGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HISTOGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HISTOGRAM_ERRORS,
};
