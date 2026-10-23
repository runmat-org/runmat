use super::parameters_output_handle::{
    MESH_INPUTS_X_Y_Z, MESH_INPUTS_X_Y_Z_PROPS, MESH_INPUTS_Z, MESH_INPUTS_Z_PROPS,
    MESH_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const MESH_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = mesh(Z)",
        inputs: &MESH_INPUTS_Z,
        outputs: &MESH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = mesh(X, Y, Z)",
        inputs: &MESH_INPUTS_X_Y_Z,
        outputs: &MESH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = mesh(Z, Name, Value, ...)",
        inputs: &MESH_INPUTS_Z_PROPS,
        outputs: &MESH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = mesh(X, Y, Z, Name, Value, ...)",
        inputs: &MESH_INPUTS_X_Y_Z_PROPS,
        outputs: &MESH_OUTPUT_HANDLE,
    },
];
