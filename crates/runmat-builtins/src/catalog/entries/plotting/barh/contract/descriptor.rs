use super::errors::BARH_ERRORS;
use super::signatures::BARH_SIGNATURES;
use crate::*;

pub const BARH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BARH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BARH_ERRORS,
};
