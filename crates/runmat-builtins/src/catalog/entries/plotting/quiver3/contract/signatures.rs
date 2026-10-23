use super::parameters_inputs_ax_x_y_z_u_v_w::{
    INPUTS_AX_X_Y_Z_U_V_W, INPUTS_X_Y_Z_U_V_W_ARGS, INPUTS_Z_U_V_W_ARGS,
};
use super::parameters_inputs_ax_z_u_v_w_args::{
    INPUTS_AX_X_Y_Z_U_V_W_ARGS, INPUTS_AX_Z_U_V_W_ARGS,
};
use super::parameters_output_handle::{
    INPUTS_AX_Z_U_V_W, INPUTS_X_Y_Z_U_V_W, INPUTS_Z_U_V_W, OUTPUT_HANDLE,
};
use crate::*;

pub(super) const QUIVER3_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = quiver3(Z, U, V, W)",
        inputs: &INPUTS_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(X, Y, Z, U, V, W)",
        inputs: &INPUTS_X_Y_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_X_Y_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, Z, U, V, W)",
        inputs: &INPUTS_AX_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, X, Y, Z, U, V, W)",
        inputs: &INPUTS_AX_X_Y_Z_U_V_W,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_AX_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = quiver3(ax, X, Y, Z, U, V, W, scaleOrStyleOrNameValue...)",
        inputs: &INPUTS_AX_X_Y_Z_U_V_W_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
];
