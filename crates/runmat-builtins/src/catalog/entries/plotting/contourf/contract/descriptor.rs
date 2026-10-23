use super::errors::CONTOURF_ERRORS;
use super::signatures::CONTOURF_SIGNATURES;
use crate::*;

pub const CONTOURF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTOURF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTOURF_ERRORS,
};
