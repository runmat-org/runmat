use super::errors::STEM_ERRORS;
use super::signatures::STEM_SIGNATURES;
use crate::*;

pub const STEM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STEM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STEM_ERRORS,
};
