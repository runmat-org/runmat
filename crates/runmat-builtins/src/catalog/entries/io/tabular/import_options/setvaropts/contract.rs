use super::super::shared::{INVALID, OPTIONS, PAIRS, SELECTION, UPDATED};
use crate::*;

pub const SETVAROPTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "updatedOpts = setvaropts(opts, nameValuePairs...)",
            inputs: &[OPTIONS, PAIRS],
            outputs: &UPDATED,
        },
        BuiltinSignatureDescriptor {
            label: "updatedOpts = setvaropts(opts, selection, nameValuePairs...)",
            inputs: &[OPTIONS, SELECTION, PAIRS],
            outputs: &UPDATED,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[INVALID],
};
