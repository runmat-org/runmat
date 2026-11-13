use super::errors::MESH_ERRORS;
use super::signatures::MESH_SIGNATURES;
use crate::*;

pub const MESH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MESH_ERRORS,
};
