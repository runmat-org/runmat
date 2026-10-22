use super::parameters::{
    SEMILOGY_INPUTS_ARGS, SEMILOGY_INPUTS_XY, SEMILOGY_INPUTS_Y, SEMILOGY_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const SEMILOGY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = semilogy(Y)",
        inputs: &SEMILOGY_INPUTS_Y,
        outputs: &SEMILOGY_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = semilogy(X, Y)",
        inputs: &SEMILOGY_INPUTS_XY,
        outputs: &SEMILOGY_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = semilogy(args...)",
        inputs: &SEMILOGY_INPUTS_ARGS,
        outputs: &SEMILOGY_OUTPUT_HANDLE,
    },
];
