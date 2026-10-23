use super::errors::FCONTOUR_ERRORS;
use super::signatures::FCONTOUR_SIGNATURES;
use crate::*;

pub const FCONTOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FCONTOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FCONTOUR_ERRORS,
};
