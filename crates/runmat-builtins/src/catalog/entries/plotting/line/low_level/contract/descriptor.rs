use super::errors::LINE_ERRORS;
use super::signatures::LINE_SIGNATURES;
use crate::*;

pub const LINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINE_ERRORS,
};
