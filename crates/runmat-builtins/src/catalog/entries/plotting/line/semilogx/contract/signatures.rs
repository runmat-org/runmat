use super::parameters::{
    SEMILOGX_INPUTS_ARGS, SEMILOGX_INPUTS_XY, SEMILOGX_INPUTS_Y, SEMILOGX_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const SEMILOGX_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = semilogx(Y)",
        inputs: &SEMILOGX_INPUTS_Y,
        outputs: &SEMILOGX_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = semilogx(X, Y)",
        inputs: &SEMILOGX_INPUTS_XY,
        outputs: &SEMILOGX_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = semilogx(args...)",
        inputs: &SEMILOGX_INPUTS_ARGS,
        outputs: &SEMILOGX_OUTPUT_HANDLE,
    },
];
