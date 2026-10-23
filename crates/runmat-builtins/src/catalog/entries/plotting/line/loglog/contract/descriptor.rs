use super::errors::LOGLOG_ERRORS;
use super::signatures::LOGLOG_SIGNATURES;
use crate::*;

pub const LOGLOG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOGLOG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOGLOG_ERRORS,
};
