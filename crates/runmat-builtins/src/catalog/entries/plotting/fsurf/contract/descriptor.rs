use super::errors::FSURF_ERRORS;
use super::signatures::FSURF_SIGNATURES;
use crate::*;

pub const FSURF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FSURF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FSURF_ERRORS,
};
