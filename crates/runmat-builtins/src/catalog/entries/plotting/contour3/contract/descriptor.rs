use super::errors::CONTOUR3_ERRORS;
use super::signatures::CONTOUR3_SIGNATURES;
use crate::*;

pub const CONTOUR3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTOUR3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTOUR3_ERRORS,
};
