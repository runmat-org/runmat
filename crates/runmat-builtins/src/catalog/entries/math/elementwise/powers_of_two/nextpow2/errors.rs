use crate::BuiltinErrorDescriptor;

pub const NEXTPOW2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEXTPOW2.INVALID_INPUT",
    identifier: Some("RunMat:nextpow2:InvalidInput"),
    when: "Input is not supported real numeric or logical data.",
    message: "nextpow2: invalid input",
};

pub const NEXTPOW2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEXTPOW2.INTERNAL",
    identifier: Some("RunMat:nextpow2:Internal"),
    when: "Tensor construction, provider execution, gather, or residency restoration fails.",
    message: "nextpow2: internal error",
};

pub const NEXTPOW2_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEXTPOW2.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:nextpow2:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "nextpow2: too many output arguments",
};
