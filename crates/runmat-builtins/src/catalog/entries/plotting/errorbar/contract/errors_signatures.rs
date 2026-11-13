use super::errors_output_handle::{
    ERRORBAR_INPUTS_AX_DATA_PROPS, ERRORBAR_INPUTS_X_Y_ERR, ERRORBAR_INPUTS_X_Y_YNEG_YPOS,
    ERRORBAR_INPUTS_X_Y_YNEG_YPOS_XNEG_XPOS, ERRORBAR_INPUTS_Y_ERR, ERRORBAR_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const ERRORBAR_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = errorbar(Y, E)",
        inputs: &ERRORBAR_INPUTS_Y_ERR,
        outputs: &ERRORBAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = errorbar(X, Y, E)",
        inputs: &ERRORBAR_INPUTS_X_Y_ERR,
        outputs: &ERRORBAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = errorbar(X, Y, YNeg, YPos)",
        inputs: &ERRORBAR_INPUTS_X_Y_YNEG_YPOS,
        outputs: &ERRORBAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = errorbar(X, Y, YNeg, YPos, XNeg, XPos)",
        inputs: &ERRORBAR_INPUTS_X_Y_YNEG_YPOS_XNEG_XPOS,
        outputs: &ERRORBAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = errorbar(ax, ...)",
        inputs: &ERRORBAR_INPUTS_AX_DATA_PROPS,
        outputs: &ERRORBAR_OUTPUT_HANDLE,
    },
];

pub const ERRORBAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERRORBAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:errorbar:InvalidArgument"),
    when: "Input vectors, error vectors, style arguments, or axes-target forms are malformed.",
    message: "errorbar: invalid argument",
};

pub(super) const ERRORBAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERRORBAR.INTERNAL",
    identifier: Some("RunMat:errorbar:Internal"),
    when: "Internal render preparation or GPU vertex generation fails.",
    message: "errorbar: internal operation failed",
};

pub(super) const ERRORBAR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERRORBAR_ERROR_INVALID_ARGUMENT,
    ERRORBAR_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];

pub const ERRORBAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERRORBAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORBAR_ERRORS,
};
