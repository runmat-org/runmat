use super::parameters_output_handle::{
    INPUT_ARGS, INPUT_THETA, INPUT_THETA_BINS, OUTPUT_HANDLE, POLARHISTOGRAM_INPUTS_MANUAL,
};
use crate::*;

pub(super) const POLARHISTOGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta)",
        inputs: &INPUT_THETA,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta, nbins)",
        inputs: &INPUT_THETA_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta, edges)",
        inputs: &INPUT_THETA_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta, Name, Value, ...)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(pax, ___)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram('BinEdges', edges, 'BinCounts', counts)",
        inputs: &POLARHISTOGRAM_INPUTS_MANUAL,
        outputs: &OUTPUT_HANDLE,
    },
];
