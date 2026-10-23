use super::errors::QUIVER_ERRORS;
use super::signatures::QUIVER_SIGNATURES;
use crate::*;

pub const QUIVER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QUIVER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QUIVER_ERRORS,
};
