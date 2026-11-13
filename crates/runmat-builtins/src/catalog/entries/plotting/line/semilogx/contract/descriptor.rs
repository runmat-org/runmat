use super::errors::SEMILOGX_ERRORS;
use super::signatures::SEMILOGX_SIGNATURES;
use crate::*;

pub const SEMILOGX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SEMILOGX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SEMILOGX_ERRORS,
};
