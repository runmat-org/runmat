use super::super::shared::{INVALID, OPTIONS, SELECTION, TYPE, UPDATED};
use crate::*;

pub const SETVARTYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "updatedOpts = setvartype(opts, selection, type)",
        inputs: &[OPTIONS, SELECTION, TYPE],
        outputs: &UPDATED,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[INVALID],
};
