use super::errors::PLOT3_ERRORS;
use super::signatures::PLOT3_SIGNATURES;
use crate::*;

pub const PLOT3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLOT3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PLOT3_ERRORS,
};
