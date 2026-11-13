use super::errors::SCATTER3_ERRORS;
use super::signatures::SCATTER3_SIGNATURES;
use crate::*;

pub const SCATTER3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SCATTER3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SCATTER3_ERRORS,
};
