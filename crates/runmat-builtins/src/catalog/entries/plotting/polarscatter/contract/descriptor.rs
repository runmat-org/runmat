use super::errors::POLARSCATTER_ERRORS;
use super::signatures::POLARSCATTER_SIGNATURES;
use crate::*;

pub const POLARSCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLARSCATTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLARSCATTER_ERRORS,
};
