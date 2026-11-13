use super::parameters_inputs_x_y_z_level_values::{
    CONTOUR_INPUTS_TARGET_ARGS, CONTOUR_INPUTS_X_Y_Z_LEVEL_PROPS, CONTOUR_INPUTS_X_Y_Z_LEVEL_VALUES,
};
use super::parameters_output_handle::{
    CONTOUR_INPUTS_X_Y_Z, CONTOUR_INPUTS_X_Y_Z_LEVEL_COUNT, CONTOUR_INPUTS_Z,
    CONTOUR_INPUTS_Z_LEVEL_COUNT, CONTOUR_INPUTS_Z_LEVEL_PROPS, CONTOUR_INPUTS_Z_LEVEL_VALUES,
    CONTOUR_INPUTS_Z_PROPS, CONTOUR_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const CONTOUR_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "h = contour(Z)",
        inputs: &CONTOUR_INPUTS_Z,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(Z, N)",
        inputs: &CONTOUR_INPUTS_Z_LEVEL_COUNT,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(Z, V)",
        inputs: &CONTOUR_INPUTS_Z_LEVEL_VALUES,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(Z, Name, Value, ...)",
        inputs: &CONTOUR_INPUTS_Z_PROPS,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(Z, V, Name, Value, ...)",
        inputs: &CONTOUR_INPUTS_Z_LEVEL_PROPS,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(X, Y, Z)",
        inputs: &CONTOUR_INPUTS_X_Y_Z,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(X, Y, Z, N)",
        inputs: &CONTOUR_INPUTS_X_Y_Z_LEVEL_COUNT,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(X, Y, Z, V)",
        inputs: &CONTOUR_INPUTS_X_Y_Z_LEVEL_VALUES,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(X, Y, Z, V, Name, Value, ...)",
        inputs: &CONTOUR_INPUTS_X_Y_Z_LEVEL_PROPS,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(X, Y, Z, N, Name, Value, ...)",
        inputs: &CONTOUR_INPUTS_X_Y_Z_LEVEL_PROPS,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contour(ax, ...)",
        inputs: &CONTOUR_INPUTS_TARGET_ARGS,
        outputs: &CONTOUR_OUTPUT_HANDLE,
    },
];
