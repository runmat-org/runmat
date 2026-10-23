use super::parameters_output_handle::{
    MESHC_INPUTS_X_Y_Z, MESHC_INPUTS_X_Y_Z_PROPS, MESHC_INPUTS_Z, MESHC_INPUTS_Z_PROPS,
    MESHC_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const MESHC_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = meshc(Z)",
        inputs: &MESHC_INPUTS_Z,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = meshc(X, Y, Z)",
        inputs: &MESHC_INPUTS_X_Y_Z,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = meshc(Z, Name, Value, ...)",
        inputs: &MESHC_INPUTS_Z_PROPS,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = meshc(X, Y, Z, Name, Value, ...)",
        inputs: &MESHC_INPUTS_X_Y_Z_PROPS,
        outputs: &MESHC_OUTPUT_HANDLE,
    },
];
