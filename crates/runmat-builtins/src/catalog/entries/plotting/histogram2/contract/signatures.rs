use super::parameters_output_handle::{
    INPUTS_AX_XY, INPUTS_AX_XY_BINS, INPUTS_AX_XY_BINXY, INPUTS_AX_XY_OPTIONS, INPUTS_XY,
    INPUTS_XY_BINS, INPUTS_XY_BINXY, INPUTS_XY_OPTIONS, OUTPUT_HANDLE,
};
use crate::*;

pub(super) const SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y)",
        inputs: &INPUTS_XY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y, bins)",
        inputs: &INPUTS_XY_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y, binsX, binsY)",
        inputs: &INPUTS_XY_BINXY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(X, Y, Name, Value, ...)",
        inputs: &INPUTS_XY_OPTIONS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y)",
        inputs: &INPUTS_AX_XY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y, bins)",
        inputs: &INPUTS_AX_XY_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y, binsX, binsY)",
        inputs: &INPUTS_AX_XY_BINXY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram2(ax, X, Y, Name, Value, ...)",
        inputs: &INPUTS_AX_XY_OPTIONS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(X, Y)",
        inputs: &INPUTS_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(X, Y, bins)",
        inputs: &INPUTS_XY_BINS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(ax, X, Y)",
        inputs: &INPUTS_AX_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "histogram2(ax, X, Y, Name, Value, ...)",
        inputs: &INPUTS_AX_XY_OPTIONS,
        outputs: &[],
    },
];
