use super::parameters_output_handle::{
    INPUT_ARGS, INPUT_X_Y, INPUT_X_Y_WIDTH, INPUT_Y, INPUT_Y_WIDTH, OUTPUT_HANDLE,
};
use crate::*;

pub(super) const RIBBON_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = ribbon(Y)",
        inputs: &INPUT_Y,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(Y, width)",
        inputs: &INPUT_Y_WIDTH,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(X, Y)",
        inputs: &INPUT_X_Y,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(X, Y, width)",
        inputs: &INPUT_X_Y_WIDTH,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(___, Name, Value, ...)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ribbon(ax, ___)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
];
