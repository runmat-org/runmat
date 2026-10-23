use super::parameters_inputs_x_y_z_level_props::{
    CONTOURF_INPUTS_TARGET_ARGS, CONTOURF_INPUTS_X_Y_Z_LEVEL_PROPS,
};
use super::parameters_output_handle::{
    CONTOURF_INPUTS_X_Y_Z, CONTOURF_INPUTS_X_Y_Z_LEVEL, CONTOURF_INPUTS_Z, CONTOURF_INPUTS_Z_LEVEL,
    CONTOURF_INPUTS_Z_LEVEL_PROPS, CONTOURF_INPUTS_Z_PROPS, CONTOURF_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const CONTOURF_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z)",
        inputs: &CONTOURF_INPUTS_Z,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z, V)",
        inputs: &CONTOURF_INPUTS_Z_LEVEL,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z, Name, Value, ...)",
        inputs: &CONTOURF_INPUTS_Z_PROPS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(Z, V, Name, Value, ...)",
        inputs: &CONTOURF_INPUTS_Z_LEVEL_PROPS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(X, Y, Z)",
        inputs: &CONTOURF_INPUTS_X_Y_Z,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(X, Y, Z, V)",
        inputs: &CONTOURF_INPUTS_X_Y_Z_LEVEL,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(X, Y, Z, V, Name, Value, ...)",
        inputs: &CONTOURF_INPUTS_X_Y_Z_LEVEL_PROPS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = contourf(ax, ...)",
        inputs: &CONTOURF_INPUTS_TARGET_ARGS,
        outputs: &CONTOURF_OUTPUT_HANDLE,
    },
];
