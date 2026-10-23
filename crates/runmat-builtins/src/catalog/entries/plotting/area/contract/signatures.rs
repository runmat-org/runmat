use super::parameters_inputs_ax_y::{
    AREA_INPUTS_AX_X_Y, AREA_INPUTS_AX_X_Y_BASE, AREA_INPUTS_AX_X_Y_PROPS, AREA_INPUTS_AX_Y,
    AREA_INPUTS_AX_Y_BASE, AREA_INPUTS_AX_Y_PROPS,
};
use super::parameters_output_handle::{
    AREA_INPUTS_X_Y, AREA_INPUTS_X_Y_BASE, AREA_INPUTS_X_Y_PROPS, AREA_INPUTS_X_Y_STYLE,
    AREA_INPUTS_Y, AREA_INPUTS_Y_BASE, AREA_INPUTS_Y_PROPS, AREA_INPUTS_Y_STYLE,
    AREA_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const AREA_SIGNATURES: [BuiltinSignatureDescriptor; 14] = [
    BuiltinSignatureDescriptor {
        label: "h = area(Y)",
        inputs: &AREA_INPUTS_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(Y, LineSpec)",
        inputs: &AREA_INPUTS_Y_STYLE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(Y, basevalue)",
        inputs: &AREA_INPUTS_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y)",
        inputs: &AREA_INPUTS_X_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y, LineSpec)",
        inputs: &AREA_INPUTS_X_Y_STYLE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y, basevalue)",
        inputs: &AREA_INPUTS_X_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(X, Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_X_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, Y)",
        inputs: &AREA_INPUTS_AX_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_AX_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, Y, basevalue)",
        inputs: &AREA_INPUTS_AX_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, X, Y)",
        inputs: &AREA_INPUTS_AX_X_Y,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, X, Y, Name, Value, ...)",
        inputs: &AREA_INPUTS_AX_X_Y_PROPS,
        outputs: &AREA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = area(ax, X, Y, basevalue)",
        inputs: &AREA_INPUTS_AX_X_Y_BASE,
        outputs: &AREA_OUTPUT_HANDLE,
    },
];
