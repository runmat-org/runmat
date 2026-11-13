use super::errors::STAIRS_ERRORS;
use super::signatures::STAIRS_SIGNATURES;
use crate::*;

pub const STAIRS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STAIRS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STAIRS_ERRORS,
};
