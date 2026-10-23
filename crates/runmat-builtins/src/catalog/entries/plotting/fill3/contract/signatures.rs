use super::parameters_output_handle::{
    FILL3_INPUTS_AX_MULTI_GROUP, FILL3_INPUTS_AX_MULTI_GROUP_PROPS, FILL3_INPUTS_AX_X_Y_Z_C,
    FILL3_INPUTS_MULTI_GROUP, FILL3_INPUTS_MULTI_GROUP_PROPS, FILL3_INPUTS_X_Y_Z_C,
    FILL3_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const FILL3_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = fill3(X, Y, Z, C)",
        inputs: &FILL3_INPUTS_X_Y_Z_C,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(X1, Y1, Z1, C1, ..., Xn, Yn, Zn, Cn)",
        inputs: &FILL3_INPUTS_MULTI_GROUP,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(X1, Y1, Z1, C1, ..., Name, Value, ...)",
        inputs: &FILL3_INPUTS_MULTI_GROUP_PROPS,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(ax, X, Y, Z, C)",
        inputs: &FILL3_INPUTS_AX_X_Y_Z_C,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(ax, X1, Y1, Z1, C1, ..., Xn, Yn, Zn, Cn)",
        inputs: &FILL3_INPUTS_AX_MULTI_GROUP,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = fill3(ax, X1, Y1, Z1, C1, ..., Name, Value, ...)",
        inputs: &FILL3_INPUTS_AX_MULTI_GROUP_PROPS,
        outputs: &FILL3_OUTPUT_HANDLE,
    },
];
