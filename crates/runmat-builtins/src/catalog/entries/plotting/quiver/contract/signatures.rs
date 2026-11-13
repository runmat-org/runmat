use super::parameters_inputs_ax_x_y_u_v::{
    QUIVER_INPUTS_AX_X_Y_U_V, QUIVER_INPUTS_AX_X_Y_U_V_PROPS, QUIVER_INPUTS_AX_X_Y_U_V_STYLE,
};
use super::parameters_output_handle::{
    QUIVER_INPUTS_AX_U_V, QUIVER_INPUTS_U_V, QUIVER_INPUTS_X_Y_U_V, QUIVER_INPUTS_X_Y_U_V_PROPS,
    QUIVER_INPUTS_X_Y_U_V_STYLE, QUIVER_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const QUIVER_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = quiver(U, V)",
        inputs: &QUIVER_INPUTS_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(X, Y, U, V)",
        inputs: &QUIVER_INPUTS_X_Y_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(X, Y, U, V, LineSpec)",
        inputs: &QUIVER_INPUTS_X_Y_U_V_STYLE,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(X, Y, U, V, Name, Value, ...)",
        inputs: &QUIVER_INPUTS_X_Y_U_V_PROPS,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, U, V)",
        inputs: &QUIVER_INPUTS_AX_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, X, Y, U, V)",
        inputs: &QUIVER_INPUTS_AX_X_Y_U_V,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, X, Y, U, V, LineSpec)",
        inputs: &QUIVER_INPUTS_AX_X_Y_U_V_STYLE,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver(ax, X, Y, U, V, Name, Value, ...)",
        inputs: &QUIVER_INPUTS_AX_X_Y_U_V_PROPS,
        outputs: &QUIVER_OUTPUT_HANDLE,
    },
];
