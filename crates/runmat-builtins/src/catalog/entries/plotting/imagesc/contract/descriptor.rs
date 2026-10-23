use super::errors::IMAGESC_ERRORS;
use super::signatures::IMAGESC_SIGNATURES;
use crate::*;

pub const IMAGESC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMAGESC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMAGESC_ERRORS,
};
