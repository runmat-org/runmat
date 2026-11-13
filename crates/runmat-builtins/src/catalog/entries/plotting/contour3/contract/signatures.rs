use super::parameters_inputs_x_y_z_level_values::{
    CONTOUR3_INPUTS_TARGET_ARGS, CONTOUR3_INPUTS_X_Y_Z_LEVEL_PROPS,
    CONTOUR3_INPUTS_X_Y_Z_LEVEL_VALUES,
};
use super::parameters_output_handle::{
    CONTOUR3_INPUTS_X_Y_Z, CONTOUR3_INPUTS_X_Y_Z_LEVEL_COUNT, CONTOUR3_INPUTS_Z,
    CONTOUR3_INPUTS_Z_LEVEL_COUNT, CONTOUR3_INPUTS_Z_LEVEL_PROPS, CONTOUR3_INPUTS_Z_LEVEL_VALUES,
    CONTOUR3_INPUTS_Z_PROPS, CONTOUR3_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const CONTOUR3_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "h = contour3(Z)",
        inputs: &CONTOUR3_INPUTS_Z,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(Z, N)",
        inputs: &CONTOUR3_INPUTS_Z_LEVEL_COUNT,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(Z, V)",
        inputs: &CONTOUR3_INPUTS_Z_LEVEL_VALUES,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(Z, Name, Value, ...)",
        inputs: &CONTOUR3_INPUTS_Z_PROPS,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(Z, V, Name, Value, ...)",
        inputs: &CONTOUR3_INPUTS_Z_LEVEL_PROPS,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(X, Y, Z)",
        inputs: &CONTOUR3_INPUTS_X_Y_Z,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(X, Y, Z, N)",
        inputs: &CONTOUR3_INPUTS_X_Y_Z_LEVEL_COUNT,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(X, Y, Z, V)",
        inputs: &CONTOUR3_INPUTS_X_Y_Z_LEVEL_VALUES,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(X, Y, Z, V, Name, Value, ...)",
        inputs: &CONTOUR3_INPUTS_X_Y_Z_LEVEL_PROPS,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(X, Y, Z, N, Name, Value, ...)",
        inputs: &CONTOUR3_INPUTS_X_Y_Z_LEVEL_PROPS,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour3(ax, ...)",
        inputs: &CONTOUR3_INPUTS_TARGET_ARGS,
        outputs: &CONTOUR3_OUTPUT_HANDLE,
    },
];
