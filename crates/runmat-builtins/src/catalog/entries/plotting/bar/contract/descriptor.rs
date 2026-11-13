use super::errors::BAR_ERRORS;
use super::signatures::BAR_SIGNATURES;
use crate::*;

pub const BAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BAR_ERRORS,
};
