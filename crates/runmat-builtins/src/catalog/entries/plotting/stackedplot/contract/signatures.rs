use super::parameters_output::{INPUTS_PROPS, INPUTS_TABLE, INPUTS_X_Y, INPUTS_Y, OUTPUT};
use crate::*;

pub(super) const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(Y)",
        inputs: &INPUTS_Y,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(tbl)",
        inputs: &INPUTS_TABLE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(tbl, vars)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(..., LineSpec)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(..., Name, Value)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(parent, ...)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
];
