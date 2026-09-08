use crate::BuiltinErrorDescriptor;

pub const CELLFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.INVALID_INPUT",
    identifier: Some("RunMat:cellfun:InvalidInput"),
    when: "Input callback, cell-array arguments, or name-value forms are invalid.",
    message: "cellfun: invalid input arguments",
};

pub const CELLFUN_ERROR_UNIFORM_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.UNIFORM_OUTPUT",
    identifier: Some("RunMat:cellfun:UniformOutput"),
    when: "UniformOutput requirements are violated.",
    message: "cellfun: uniform output contract violated",
};

pub const CELLFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:cellfun:FunctionError"),
    when: "The mapped callback or ErrorHandler cannot complete.",
    message: "cellfun: callback execution error",
};

pub const CELLFUN_ERROR_UNDEFINED_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.UNDEFINED_FUNCTION",
    identifier: Some("RunMat:UndefinedFunction"),
    when: "External callable resolution fails at the runtime boundary.",
    message: "cellfun: undefined external function",
};

pub const CELLFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.INTERNAL",
    identifier: None,
    when: "Internal allocation or result materialization fails.",
    message: "cellfun: internal error",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    CELLFUN_ERROR_INVALID_INPUT,
    CELLFUN_ERROR_UNIFORM_OUTPUT,
    CELLFUN_ERROR_FUNCTION_ERROR,
    CELLFUN_ERROR_UNDEFINED_FUNCTION,
    CELLFUN_ERROR_INTERNAL,
];
