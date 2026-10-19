use crate::BuiltinErrorDescriptor;

pub const POW2_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.INVALID_ARGUMENT",
    identifier: Some("RunMat:pow2:InvalidArgument"),
    when: "The call has neither one nor two inputs.",
    message: "pow2: invalid argument",
};
pub const POW2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.INVALID_INPUT",
    identifier: Some("RunMat:pow2:InvalidInput"),
    when: "An input is not a supported numeric, logical, or character value.",
    message: "pow2: invalid input",
};
pub const POW2_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.SIZE_MISMATCH",
    identifier: Some("RunMat:pow2:SizeMismatch"),
    when: "The two-input form receives operands that cannot be implicitly expanded.",
    message: "pow2: size mismatch",
};
pub const POW2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.INTERNAL",
    identifier: Some("RunMat:pow2:Internal"),
    when: "Conversion, allocation, provider execution, gather, or restoration fails.",
    message: "pow2: internal error",
};
pub const POW2_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POW2.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pow2:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "pow2: too many output arguments",
};
