use super::errors::IMSHOW_ERRORS;
use super::signatures::IMSHOW_SIGNATURES;
use crate::*;

pub const IMSHOW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMSHOW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMSHOW_ERRORS,
};
