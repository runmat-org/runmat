use super::errors::QUIVER3_ERRORS;
use super::signatures::QUIVER3_SIGNATURES;
use crate::*;

pub const QUIVER3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QUIVER3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QUIVER3_ERRORS,
};
