//! Structured least-squares-line signatures and diagnostic ownership.
use crate::*;

const AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};
const HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line graphics handle or vector of line handles.",
};
const EMPTY: [BuiltinParamDescriptor; 0] = [];
const AX_INPUT: [BuiltinParamDescriptor; 1] = [AX];
const HANDLE_OUTPUT: [BuiltinParamDescriptor; 1] = [HANDLE];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "lsline()",
        inputs: &EMPTY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "lsline(ax)",
        inputs: &AX_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = lsline()",
        inputs: &EMPTY,
        outputs: &HANDLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = lsline(ax)",
        inputs: &AX_INPUT,
        outputs: &HANDLE_OUTPUT,
    },
];
pub const LSLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsline:InvalidArgument"),
    when: "The optional axes handle is malformed or extra arguments are supplied.",
    message: "lsline: invalid argument",
};
pub const LSLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSLINE.INTERNAL",
    identifier: Some("RunMat:lsline:Internal"),
    when: "RunMat cannot construct or register the least-squares line.",
    message: "lsline: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [LSLINE_ERROR_INVALID_ARGUMENT, LSLINE_ERROR_INTERNAL];
pub const LSLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
