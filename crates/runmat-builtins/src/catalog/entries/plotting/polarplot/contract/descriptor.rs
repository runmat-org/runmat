use super::errors::POLARPLOT_ERRORS;
use super::signatures::POLARPLOT_SIGNATURES;
use crate::*;

pub const POLARPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLARPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLARPLOT_ERRORS,
};
