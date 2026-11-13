use super::parameters_output_handle::{
    INPUTS_AX_TRI_X_Y, INPUTS_AX_TR_STYLE, INPUTS_TR, INPUTS_TRI_X_Y, INPUTS_TRI_X_Y_PROPS,
    INPUTS_TRI_X_Y_STYLE, INPUTS_TR_STYLE, TRIPLOT_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const TRIPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "h = triplot(TRI, X, Y)",
        inputs: &INPUTS_TRI_X_Y,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = triplot(TRI, X, Y, LineSpec)",
        inputs: &INPUTS_TRI_X_Y_STYLE,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = triplot(TRI, X, Y, Name, Value, ...)",
        inputs: &INPUTS_TRI_X_Y_PROPS,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = triplot(ax, TRI, X, Y)",
        inputs: &INPUTS_AX_TRI_X_Y,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = triplot(TR)",
        inputs: &INPUTS_TR,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = triplot(TR, LineSpec)",
        inputs: &INPUTS_TR_STYLE,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = triplot(ax, TR, ...)",
        inputs: &INPUTS_AX_TR_STYLE,
        outputs: &TRIPLOT_OUTPUT_HANDLE,
    },
];
