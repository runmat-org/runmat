use super::errors::SURFC_ERRORS;
use super::signatures::SURFC_SIGNATURES;
use crate::*;

pub const SURFC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SURFC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SURFC_ERRORS,
};
