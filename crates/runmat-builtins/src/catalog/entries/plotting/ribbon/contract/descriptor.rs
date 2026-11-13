use super::errors::RIBBON_ERRORS;
use super::signatures::RIBBON_SIGNATURES;
use crate::*;

pub const RIBBON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RIBBON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RIBBON_ERRORS,
};
