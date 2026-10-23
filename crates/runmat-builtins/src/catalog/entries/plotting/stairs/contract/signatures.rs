use super::parameters_inputs_ax_x_y_props::STAIRS_INPUTS_AX_X_Y_PROPS;
use super::parameters_output_handle::{
    STAIRS_INPUTS_AX_X_Y, STAIRS_INPUTS_AX_X_Y_STYLE, STAIRS_INPUTS_AX_Y, STAIRS_INPUTS_X_Y,
    STAIRS_INPUTS_X_Y_PROPS, STAIRS_INPUTS_X_Y_STYLE, STAIRS_INPUTS_Y, STAIRS_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const STAIRS_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = stairs(Y)",
        inputs: &STAIRS_INPUTS_Y,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(X, Y)",
        inputs: &STAIRS_INPUTS_X_Y,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(X, Y, LineSpec)",
        inputs: &STAIRS_INPUTS_X_Y_STYLE,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(X, Y, Name, Value, ...)",
        inputs: &STAIRS_INPUTS_X_Y_PROPS,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(ax, Y)",
        inputs: &STAIRS_INPUTS_AX_Y,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(ax, X, Y)",
        inputs: &STAIRS_INPUTS_AX_X_Y,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(ax, X, Y, LineSpec)",
        inputs: &STAIRS_INPUTS_AX_X_Y_STYLE,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stairs(ax, X, Y, Name, Value, ...)",
        inputs: &STAIRS_INPUTS_AX_X_Y_PROPS,
        outputs: &STAIRS_OUTPUT_HANDLE,
    },
];
