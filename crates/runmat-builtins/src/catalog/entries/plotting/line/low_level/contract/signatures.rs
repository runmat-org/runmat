use super::parameters::{
    LINE_INPUTS_PROPS, LINE_INPUTS_X_Y, LINE_INPUTS_X_Y_Z, LINE_OUTPUT_HANDLES,
};
use crate::*;

pub(super) const LINE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = line()",
        inputs: &[],
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(X, Y)",
        inputs: &LINE_INPUTS_X_Y,
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(X, Y, Z)",
        inputs: &LINE_INPUTS_X_Y_Z,
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(X, Y, Name, Value, ...)",
        inputs: &LINE_INPUTS_PROPS,
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(Name, Value, ...)",
        inputs: &LINE_INPUTS_PROPS,
        outputs: &LINE_OUTPUT_HANDLES,
    },
];
