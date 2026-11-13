use super::errors::FILL3_ERRORS;
use super::signatures::FILL3_SIGNATURES;
use crate::*;

pub const FILL3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILL3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILL3_ERRORS,
};
