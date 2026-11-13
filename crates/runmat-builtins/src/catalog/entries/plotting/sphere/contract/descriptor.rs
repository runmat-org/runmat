use super::errors::SPHERE_ERRORS;
use super::signatures::SPHERE_SIGNATURES;
use crate::*;

pub const SPHERE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPHERE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPHERE_ERRORS,
};
