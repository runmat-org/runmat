use super::errors::SEMILOGY_ERRORS;
use super::signatures::SEMILOGY_SIGNATURES;
use crate::*;

pub const SEMILOGY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SEMILOGY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SEMILOGY_ERRORS,
};
