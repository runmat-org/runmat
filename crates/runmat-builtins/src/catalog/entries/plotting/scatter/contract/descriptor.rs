use super::errors::SCATTER_ERRORS;
use super::signatures::SCATTER_SIGNATURES;
use crate::*;

pub const SCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SCATTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SCATTER_ERRORS,
};
