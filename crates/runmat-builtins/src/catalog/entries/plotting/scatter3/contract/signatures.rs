use super::parameters_inputs_ax_x_y_z::{
    SCATTER3_INPUTS_AX_X_Y_Z, SCATTER3_INPUTS_AX_X_Y_Z_S, SCATTER3_INPUTS_AX_X_Y_Z_STYLE,
    SCATTER3_INPUTS_AX_X_Y_Z_S_C,
};
use super::parameters_inputs_ax_x_y_z_props::SCATTER3_INPUTS_AX_X_Y_Z_PROPS;
use super::parameters_output_handle::{
    SCATTER3_INPUTS_X_Y_Z, SCATTER3_INPUTS_X_Y_Z_PROPS, SCATTER3_INPUTS_X_Y_Z_S,
    SCATTER3_INPUTS_X_Y_Z_STYLE, SCATTER3_INPUTS_X_Y_Z_S_C, SCATTER3_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const SCATTER3_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z)",
        inputs: &SCATTER3_INPUTS_X_Y_Z,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, S)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_S,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, S, C)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_S_C,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, LineSpec)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_STYLE,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(X, Y, Z, Name, Value, ...)",
        inputs: &SCATTER3_INPUTS_X_Y_Z_PROPS,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, S)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_S,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, S, C)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_S_C,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, LineSpec)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_STYLE,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter3(ax, X, Y, Z, Name, Value, ...)",
        inputs: &SCATTER3_INPUTS_AX_X_Y_Z_PROPS,
        outputs: &SCATTER3_OUTPUT_HANDLE,
    },
];
