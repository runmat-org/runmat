use super::super::shared::{INVALID, OPTIONS, SELECTION, VARIABLE};
use crate::*;

pub const GETVAROPTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "variableOpts = getvaropts(opts, selection)",
        inputs: &[OPTIONS, SELECTION],
        outputs: &VARIABLE,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[INVALID],
};
