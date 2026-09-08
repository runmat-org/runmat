use crate::BuiltinErrorDescriptor;

macro_rules! error {
    ($name:ident, $code:literal, $identifier:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            when: $when,
            message: $message,
        };
    };
}

error!(
    ARRAYFUN_ERROR_INVALID_INPUT,
    "RM.ARRAYFUN.INVALID_INPUT",
    "RunMat:arrayfun:InvalidInput",
    "The callable, arrays, or option tail violates the arrayfun input contract.",
    "arrayfun: invalid input arguments"
);
error!(
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION,
    "RM.ARRAYFUN.UNIFORM_OUTPUT_OPTION",
    "RunMat:arrayfun:UniformOutputOption",
    "UniformOutput is not logical true or false, or double one or zero.",
    "arrayfun: UniformOutput must be logical true or false"
);
error!(
    ARRAYFUN_ERROR_CALLBACK_FAILED,
    "RM.ARRAYFUN.CALLBACK_FAILED",
    "RunMat:arrayfun:CallbackFailed",
    "A callback fails and no ErrorHandler recovers the element.",
    "arrayfun: callback execution failed"
);
error!(
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
    "RM.ARRAYFUN.UNIFORM_OUTPUT_TYPE",
    "RunMat:arrayfun:UniformOutputType",
    "A uniform callback result is nonscalar or differs in class from another result.",
    "arrayfun: callback must return scalar values of one class for UniformOutput=true"
);
error!(
    ARRAYFUN_ERROR_INTERNAL,
    "RM.ARRAYFUN.INTERNAL",
    "RunMat:arrayfun:InternalError",
    "Internal planning, materialization, collection, or provider transfer fails.",
    "arrayfun: internal error"
);
error!(
    ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
    "RM.ARRAYFUN.UNDEFINED_FUNCTION",
    "RunMat:UndefinedFunction",
    "The callable identity cannot be resolved at execution.",
    "arrayfun: undefined function"
);

pub(super) const ERRORS: [BuiltinErrorDescriptor; 6] = [
    ARRAYFUN_ERROR_INVALID_INPUT,
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION,
    ARRAYFUN_ERROR_CALLBACK_FAILED,
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
    ARRAYFUN_ERROR_INTERNAL,
    ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
];
