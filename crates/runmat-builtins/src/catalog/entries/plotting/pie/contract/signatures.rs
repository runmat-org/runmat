use super::parameters_output_handle::{
    PIE_INPUTS_AX_DATA, PIE_INPUTS_X, PIE_INPUTS_X_EXPLODE, PIE_INPUTS_X_EXPLODE_LABELS,
    PIE_INPUTS_X_LABELS, PIE_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const PIE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = pie(X)",
        inputs: &PIE_INPUTS_X,
        outputs: &PIE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = pie(X, explode)",
        inputs: &PIE_INPUTS_X_EXPLODE,
        outputs: &PIE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = pie(X, labels)",
        inputs: &PIE_INPUTS_X_LABELS,
        outputs: &PIE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = pie(X, explode, labels)",
        inputs: &PIE_INPUTS_X_EXPLODE_LABELS,
        outputs: &PIE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = pie(ax, ...)",
        inputs: &PIE_INPUTS_AX_DATA,
        outputs: &PIE_OUTPUT_HANDLE,
    },
];
