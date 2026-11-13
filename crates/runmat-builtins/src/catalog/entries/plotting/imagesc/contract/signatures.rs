use super::parameters_output_handle::{
    IMAGESC_INPUTS_C, IMAGESC_INPUTS_C_CLIMS, IMAGESC_INPUTS_C_PROPS, IMAGESC_INPUTS_X_Y_C,
    IMAGESC_INPUTS_X_Y_C_PROPS, IMAGESC_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const IMAGESC_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = imagesc(C)",
        inputs: &IMAGESC_INPUTS_C,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(X, Y, C)",
        inputs: &IMAGESC_INPUTS_X_Y_C,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(C, clims)",
        inputs: &IMAGESC_INPUTS_C_CLIMS,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(C, Name, Value, ...)",
        inputs: &IMAGESC_INPUTS_C_PROPS,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imagesc(X, Y, C, Name, Value, ...)",
        inputs: &IMAGESC_INPUTS_X_Y_C_PROPS,
        outputs: &IMAGESC_OUTPUT_HANDLE,
    },
];
