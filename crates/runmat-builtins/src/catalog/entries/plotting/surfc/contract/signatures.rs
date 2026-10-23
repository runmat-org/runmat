use super::parameters_output_handle::{
    SURFC_INPUTS_X_Y_Z, SURFC_INPUTS_X_Y_Z_PROPS, SURFC_INPUTS_Z, SURFC_INPUTS_Z_PROPS,
    SURFC_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const SURFC_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = surfc(Z)",
        inputs: &SURFC_INPUTS_Z,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surfc(X, Y, Z)",
        inputs: &SURFC_INPUTS_X_Y_Z,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surfc(Z, Name, Value, ...)",
        inputs: &SURFC_INPUTS_Z_PROPS,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surfc(X, Y, Z, Name, Value, ...)",
        inputs: &SURFC_INPUTS_X_Y_Z_PROPS,
        outputs: &SURFC_OUTPUT_HANDLE,
    },
];
