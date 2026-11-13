use super::parameters_output_handle::{
    IMAGE_INPUTS_C, IMAGE_INPUTS_C_PROPS, IMAGE_INPUTS_X_Y_C, IMAGE_INPUTS_X_Y_C_PROPS,
    IMAGE_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const IMAGE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = image(C)",
        inputs: &IMAGE_INPUTS_C,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = image(X, Y, C)",
        inputs: &IMAGE_INPUTS_X_Y_C,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = image(C, Name, Value, ...)",
        inputs: &IMAGE_INPUTS_C_PROPS,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = image(X, Y, C, Name, Value, ...)",
        inputs: &IMAGE_INPUTS_X_Y_C_PROPS,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
];
