use super::errors::AREA_ERRORS;
use super::signatures::AREA_SIGNATURES;
use crate::*;

pub const AREA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AREA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AREA_ERRORS,
};
