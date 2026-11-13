use super::parameters::{
    LOGLOG_INPUTS_ARGS, LOGLOG_INPUTS_XY, LOGLOG_INPUTS_Y, LOGLOG_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const LOGLOG_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = loglog(Y)",
        inputs: &LOGLOG_INPUTS_Y,
        outputs: &LOGLOG_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = loglog(X, Y)",
        inputs: &LOGLOG_INPUTS_XY,
        outputs: &LOGLOG_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = loglog(args...)",
        inputs: &LOGLOG_INPUTS_ARGS,
        outputs: &LOGLOG_OUTPUT_HANDLE,
    },
];
