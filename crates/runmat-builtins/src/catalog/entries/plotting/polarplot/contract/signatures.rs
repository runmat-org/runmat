use super::parameters_output_handle::{
    POLARPLOT_INPUTS_ARGS, POLARPLOT_INPUTS_RHO, POLARPLOT_INPUTS_THETA_RHO,
    POLARPLOT_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const POLARPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = polarplot(theta, rho)",
        inputs: &POLARPLOT_INPUTS_THETA_RHO,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(theta, rho, LineSpec)",
        inputs: &POLARPLOT_INPUTS_ARGS,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(theta1, rho1, ..., thetaN, rhoN)",
        inputs: &POLARPLOT_INPUTS_ARGS,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(rho)",
        inputs: &POLARPLOT_INPUTS_RHO,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarplot(Z)",
        inputs: &POLARPLOT_INPUTS_RHO,
        outputs: &POLARPLOT_OUTPUT_HANDLE,
    },
];
