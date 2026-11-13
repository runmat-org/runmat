use super::parameters_outputs::{INPUTS, OUTPUTS};
use crate::*;

pub(super) const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = fill(X, Y, C, ..., Name, Value)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
