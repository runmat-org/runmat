use super::errors::POLARHISTOGRAM_ERRORS;
use super::signatures::POLARHISTOGRAM_SIGNATURES;
use crate::*;

pub const POLARHISTOGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLARHISTOGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLARHISTOGRAM_ERRORS,
};
