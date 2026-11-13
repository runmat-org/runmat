use super::parameters_output_handle::{
    INPUT_ARGS, INPUT_THETA_RHO, INPUT_THETA_RHO_SZ, INPUT_THETA_RHO_SZ_C,
    POLARSCATTER_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const POLARSCATTER_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(theta, rho)",
        inputs: &INPUT_THETA_RHO,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(theta, rho, sz)",
        inputs: &INPUT_THETA_RHO_SZ,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(theta, rho, sz, c)",
        inputs: &INPUT_THETA_RHO_SZ_C,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(___, mkr)",
        inputs: &INPUT_ARGS,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(___, 'filled', Name, Value)",
        inputs: &INPUT_ARGS,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ps = polarscatter(pax, ___)",
        inputs: &INPUT_ARGS,
        outputs: &POLARSCATTER_OUTPUT_HANDLE,
    },
];
