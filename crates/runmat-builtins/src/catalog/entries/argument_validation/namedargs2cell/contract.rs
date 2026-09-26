use crate::*;
pub const NAMEDARGS2CELL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "namedargs2cell accepts only a scalar structure. A top-level integer host or resident value rejects without conversion or provider access, while integer values stored in valid structure fields are preserved exactly as ordinary payloads." };
pub const NAMEDARGS2CELL_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAMEDARGS2CELL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:namedargs2cell:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "namedargs2cell: too many output arguments",
};
pub const NAMEDARGS2CELL_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAMEDARGS2CELL.INVALID_ARGUMENT",
    identifier: Some("RunMat:namedargs2cell:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "namedargs2cell: invalid argument",
};
pub const NAMEDARGS2CELL_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAMEDARGS2CELL.VALIDATION_FAILED",
    identifier: Some("RunMat:namedargs2cell:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "namedargs2cell: value does not satisfy validator",
};
pub const NAMEDARGS2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "namedargs2cell(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[super::super::support::CELL_OUTPUT],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[NAMEDARGS2CELL_ERROR_OUTPUTS, NAMEDARGS2CELL_ERROR_INVALID],
};
