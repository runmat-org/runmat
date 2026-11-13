use super::parameters_output_handle::{
    FSURF_INPUTS_AX_PROPS, FSURF_INPUTS_F, FSURF_INPUTS_F_DOMAIN, FSURF_INPUTS_PARAMETRIC,
    FSURF_INPUTS_PROPS, FSURF_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const FSURF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = fsurf(f)",
        inputs: &FSURF_INPUTS_F,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(f, xyinterval)",
        inputs: &FSURF_INPUTS_F_DOMAIN,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(fx, fy, fz)",
        inputs: &FSURF_INPUTS_PARAMETRIC,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(fx, fy, fz, uvinterval)",
        inputs: &FSURF_INPUTS_PROPS,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(___, Name, Value, ...)",
        inputs: &FSURF_INPUTS_PROPS,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fsurf(ax, ___)",
        inputs: &FSURF_INPUTS_AX_PROPS,
        outputs: &FSURF_OUTPUT_HANDLE,
    },
];
