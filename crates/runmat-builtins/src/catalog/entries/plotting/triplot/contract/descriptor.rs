use super::errors::TRIPLOT_ERRORS;
use super::signatures::TRIPLOT_SIGNATURES;
use crate::*;

pub const TRIPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRIPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRIPLOT_ERRORS,
};
