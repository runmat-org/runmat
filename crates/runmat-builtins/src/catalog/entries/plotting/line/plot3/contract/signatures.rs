use super::parameters_inputs_ax_x_y_z_props::{
    PLOT3_INPUTS_AX_MULTI_SERIES, PLOT3_INPUTS_AX_X_Y_Z_PROPS, PLOT3_INPUTS_MULTI_SERIES,
};
use super::parameters_output_handle::{
    PLOT3_INPUTS_AX_X_Y_Z, PLOT3_INPUTS_AX_X_Y_Z_STYLE, PLOT3_INPUTS_X_Y_Z,
    PLOT3_INPUTS_X_Y_Z_PROPS, PLOT3_INPUTS_X_Y_Z_STYLE, PLOT3_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const PLOT3_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = plot3(X, Y, Z)",
        inputs: &PLOT3_INPUTS_X_Y_Z,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(X, Y, Z, LineSpec)",
        inputs: &PLOT3_INPUTS_X_Y_Z_STYLE,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(X, Y, Z, Name, Value, ...)",
        inputs: &PLOT3_INPUTS_X_Y_Z_PROPS,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(ax, X, Y, Z)",
        inputs: &PLOT3_INPUTS_AX_X_Y_Z,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(ax, X, Y, Z, LineSpec)",
        inputs: &PLOT3_INPUTS_AX_X_Y_Z_STYLE,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(ax, X, Y, Z, Name, Value, ...)",
        inputs: &PLOT3_INPUTS_AX_X_Y_Z_PROPS,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(X1, Y1, Z1, ..., Xn, Yn, Zn)",
        inputs: &PLOT3_INPUTS_MULTI_SERIES,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = plot3(ax, X1, Y1, Z1, ..., Xn, Yn, Zn)",
        inputs: &PLOT3_INPUTS_AX_MULTI_SERIES,
        outputs: &PLOT3_OUTPUT_HANDLE,
    },
];
