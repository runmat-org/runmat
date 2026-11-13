use super::parameters_inputs_ax_x_y_s::{
    SCATTER_INPUTS_AX_X_Y_PROPS, SCATTER_INPUTS_AX_X_Y_S, SCATTER_INPUTS_AX_X_Y_STYLE,
    SCATTER_INPUTS_AX_X_Y_S_C,
};
use super::parameters_output_handle::{
    SCATTER_INPUTS_AX_X_Y, SCATTER_INPUTS_X_Y, SCATTER_INPUTS_X_Y_PROPS, SCATTER_INPUTS_X_Y_S,
    SCATTER_INPUTS_X_Y_STYLE, SCATTER_INPUTS_X_Y_S_C, SCATTER_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const SCATTER_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = scatter(X, Y)",
        inputs: &SCATTER_INPUTS_X_Y,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(X, Y, S)",
        inputs: &SCATTER_INPUTS_X_Y_S,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(X, Y, S, C)",
        inputs: &SCATTER_INPUTS_X_Y_S_C,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(X, Y, LineSpec)",
        inputs: &SCATTER_INPUTS_X_Y_STYLE,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(X, Y, Name, Value, ...)",
        inputs: &SCATTER_INPUTS_X_Y_PROPS,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(ax, X, Y)",
        inputs: &SCATTER_INPUTS_AX_X_Y,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(ax, X, Y, S)",
        inputs: &SCATTER_INPUTS_AX_X_Y_S,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(ax, X, Y, S, C)",
        inputs: &SCATTER_INPUTS_AX_X_Y_S_C,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(ax, X, Y, LineSpec)",
        inputs: &SCATTER_INPUTS_AX_X_Y_STYLE,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = scatter(ax, X, Y, Name, Value, ...)",
        inputs: &SCATTER_INPUTS_AX_X_Y_PROPS,
        outputs: &SCATTER_OUTPUT_HANDLE,
    },
];
