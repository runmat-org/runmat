use super::errors::PIE_ERRORS;
use super::signatures::PIE_SIGNATURES;
use crate::*;

pub const PIE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PIE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PIE_ERRORS,
};
