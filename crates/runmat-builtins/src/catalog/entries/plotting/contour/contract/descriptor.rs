use super::errors::CONTOUR_ERRORS;
use super::signatures::CONTOUR_SIGNATURES;
use crate::*;

pub const CONTOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTOUR_ERRORS,
};
