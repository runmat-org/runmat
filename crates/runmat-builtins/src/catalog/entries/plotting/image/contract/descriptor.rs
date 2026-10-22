use super::errors::IMAGE_ERRORS;
use super::signatures::IMAGE_SIGNATURES;
use crate::*;

pub const IMAGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMAGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMAGE_ERRORS,
};
