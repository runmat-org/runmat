use crate::*;

const ARG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filenameOrState",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Log filename, `on`, or `off`.",
};
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "diary",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "diary(filename)",
        inputs: &[ARG],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "diary off",
        inputs: &[ARG],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "diary on",
        inputs: &[ARG],
        outputs: &[],
    },
];
pub const DIARY_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.ARG_COUNT",
    identifier: None,
    when: "More than one input argument is passed to diary.",
    message: "diary: expected zero or one input argument",
};
pub const DIARY_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.ARG_TYPE",
    identifier: None,
    when: "The diary filename/state argument is not a nonempty string scalar or character row.",
    message: "diary: expected filename, 'on', or 'off'",
};
pub const DIARY_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.IO", identifier: Some("RunMat:diary:IO"),
    when: "The selected diary file cannot be opened for append, or an active diary write fails through its captured filesystem provider.",
    message: "diary: file I/O failed",
};
pub const DIARY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:diary:TooManyOutputs"),
    when: "One or more output arguments are requested from diary.",
    message: "diary: expected no output arguments",
};
pub const DIARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        DIARY_ERROR_ARG_COUNT,
        DIARY_ERROR_ARG_TYPE,
        DIARY_ERROR_IO,
        DIARY_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const DIARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "diary accepts text control or no input and writes console text; it does not convert numeric or integer arrays.",
};
