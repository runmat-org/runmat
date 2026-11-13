use super::errors::MESHC_ERRORS;
use super::signatures::MESHC_SIGNATURES;
use crate::*;

pub const MESHC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESHC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MESHC_ERRORS,
};
