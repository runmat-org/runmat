use super::parameters_output_handle::{
    FCONTOUR_INPUTS_AX_PROPS, FCONTOUR_INPUTS_F, FCONTOUR_INPUTS_F_DOMAIN, FCONTOUR_INPUTS_PROPS,
    FCONTOUR_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const FCONTOUR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = fcontour(f)",
        inputs: &FCONTOUR_INPUTS_F,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(f, xyinterval)",
        inputs: &FCONTOUR_INPUTS_F_DOMAIN,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(___, Name, Value, ...)",
        inputs: &FCONTOUR_INPUTS_PROPS,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fcontour(ax, ___)",
        inputs: &FCONTOUR_INPUTS_AX_PROPS,
        outputs: &FCONTOUR_OUTPUT_HANDLE,
    },
];
