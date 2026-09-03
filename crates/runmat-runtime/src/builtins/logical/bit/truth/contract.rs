use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, LogicalBinaryOperator, LogicalUnaryOperator,
};

pub(super) struct BinaryContract {
    pub(super) name: &'static str,
    pub(super) invalid: &'static BuiltinErrorDescriptor,
    pub(super) mismatch: &'static BuiltinErrorDescriptor,
    pub(super) complex_extension: Option<&'static BuiltinExtensionDescriptor>,
    pub(super) character_extension: Option<&'static BuiltinExtensionDescriptor>,
}

pub(super) fn binary(operation: LogicalBinaryOperator) -> BinaryContract {
    match operation {
        LogicalBinaryOperator::And => BinaryContract {
            name: "and",
            invalid: &runmat_builtins::AND_ERROR_INVALID_INPUT,
            mismatch: &runmat_builtins::AND_ERROR_SIZE_MISMATCH,
            complex_extension: Some(&runmat_builtins::AND_COMPLEX_INPUT_EXTENSION),
            character_extension: Some(&runmat_builtins::AND_CHARACTER_INPUT_EXTENSION),
        },
        LogicalBinaryOperator::Or => BinaryContract {
            name: "or",
            invalid: &runmat_builtins::OR_ERROR_INVALID_INPUT,
            mismatch: &runmat_builtins::OR_ERROR_SIZE_MISMATCH,
            complex_extension: Some(&runmat_builtins::OR_COMPLEX_INPUT_EXTENSION),
            character_extension: Some(&runmat_builtins::OR_CHARACTER_INPUT_EXTENSION),
        },
        LogicalBinaryOperator::Xor => BinaryContract {
            name: "xor",
            invalid: &runmat_builtins::XOR_ERROR_INVALID_INPUT,
            mismatch: &runmat_builtins::XOR_ERROR_SIZE_MISMATCH,
            complex_extension: Some(&runmat_builtins::XOR_COMPLEX_INPUT_EXTENSION),
            character_extension: None,
        },
    }
}

pub(super) fn unary(
    operation: LogicalUnaryOperator,
) -> (&'static str, &'static BuiltinErrorDescriptor) {
    match operation {
        LogicalUnaryOperator::Not => ("not", &runmat_builtins::NOT_ERROR_INVALID_INPUT),
    }
}
