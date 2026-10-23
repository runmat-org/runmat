use super::parameters_param_s::{
    INPUT_LINE_SPEC, INPUT_X, INPUT_X_LINE_SPEC, INPUT_X_Y, OUTPUT_MULTI, OUTPUT_S, OUTPUT_S_AX,
    OUTPUT_S_AX_BIGAX,
};
use crate::*;

pub(super) const SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X, LineSpec)",
        inputs: &INPUT_X_LINE_SPEC,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X, Y)",
        inputs: &INPUT_X_Y,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(X, Y, LineSpec)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = plotmatrix(ax, ...)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "[S, AX] = plotmatrix(...)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S_AX,
    },
    BuiltinSignatureDescriptor {
        label: "[S, AX, BigAx] = plotmatrix(...)",
        inputs: &INPUT_LINE_SPEC,
        outputs: &OUTPUT_S_AX_BIGAX,
    },
    BuiltinSignatureDescriptor {
        label: "[S, AX, BigAx, H, HAx] = plotmatrix(X)",
        inputs: &INPUT_X,
        outputs: &OUTPUT_MULTI,
    },
];
