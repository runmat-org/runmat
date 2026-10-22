use super::parameters_inputs_ax_y_props::{
    STEM_INPUTS_AX_X_Y, STEM_INPUTS_AX_X_Y_PROPS, STEM_INPUTS_AX_X_Y_STYLE, STEM_INPUTS_AX_Y_PROPS,
};
use super::parameters_output_handle::{
    STEM_INPUTS_AX_Y, STEM_INPUTS_AX_Y_STYLE, STEM_INPUTS_X_Y, STEM_INPUTS_X_Y_PROPS,
    STEM_INPUTS_X_Y_STYLE, STEM_INPUTS_Y, STEM_INPUTS_Y_PROPS, STEM_INPUTS_Y_STYLE,
    STEM_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const STEM_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "h = stem(Y)",
        inputs: &STEM_INPUTS_Y,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(Y, LineSpec)",
        inputs: &STEM_INPUTS_Y_STYLE,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(Y, Name, Value, ...)",
        inputs: &STEM_INPUTS_Y_PROPS,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(X, Y)",
        inputs: &STEM_INPUTS_X_Y,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(X, Y, LineSpec)",
        inputs: &STEM_INPUTS_X_Y_STYLE,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(X, Y, Name, Value, ...)",
        inputs: &STEM_INPUTS_X_Y_PROPS,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(ax, Y)",
        inputs: &STEM_INPUTS_AX_Y,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(ax, Y, LineSpec)",
        inputs: &STEM_INPUTS_AX_Y_STYLE,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(ax, Y, Name, Value, ...)",
        inputs: &STEM_INPUTS_AX_Y_PROPS,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(ax, X, Y)",
        inputs: &STEM_INPUTS_AX_X_Y,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(ax, X, Y, LineSpec)",
        inputs: &STEM_INPUTS_AX_X_Y_STYLE,
        outputs: &STEM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = stem(ax, X, Y, Name, Value, ...)",
        inputs: &STEM_INPUTS_AX_X_Y_PROPS,
        outputs: &STEM_OUTPUT_HANDLE,
    },
];
