use crate::BuiltinErrorDescriptor;

pub const STRUCTFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.INVALID_INPUT",
    identifier: Some("RunMat:structfun:InvalidInput"),
    when: "Input callback, struct argument, or name-value form is invalid.",
    message: "structfun: invalid input arguments",
};
pub const STRUCTFUN_ERROR_NOT_SCALAR_STRUCT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.NOT_SCALAR_STRUCT",
    identifier: Some("RunMat:structfun:NotScalarStruct"),
    when: "The second argument is not a scalar struct.",
    message: "structfun: second input must be a scalar struct",
};
pub const STRUCTFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.INTERNAL",
    identifier: Some("RunMat:structfun:Internal"),
    when: "Internal output materialization fails.",
    message: "structfun: internal error",
};
pub const STRUCTFUN_ERROR_UNIFORM_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.UNIFORM_OUTPUT",
    identifier: Some("RunMat:structfun:UniformOutput"),
    when: "UniformOutput requirements are violated.",
    message: "structfun: uniform output contract violated",
};
pub const STRUCTFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:structfun:FunctionError"),
    when: "The mapped callback or ErrorHandler cannot complete with the requested outputs.",
    message: "structfun: callback execution error",
};
pub const STRUCTFUN_ERROR_UNDEFINED_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRUCTFUN.UNDEFINED_FUNCTION",
    identifier: Some("RunMat:UndefinedFunction"),
    when: "External callable resolution fails at the runtime boundary.",
    message: "structfun: undefined external function",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    STRUCTFUN_ERROR_INVALID_INPUT,
    STRUCTFUN_ERROR_NOT_SCALAR_STRUCT,
    STRUCTFUN_ERROR_INTERNAL,
    STRUCTFUN_ERROR_UNIFORM_OUTPUT,
    STRUCTFUN_ERROR_FUNCTION_ERROR,
    STRUCTFUN_ERROR_UNDEFINED_FUNCTION,
];
