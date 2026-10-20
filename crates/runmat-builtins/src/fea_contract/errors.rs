//! Shared errors of the typed FEA public and method surface.

use crate::*;

pub const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INVALID_INPUT",
    identifier: Some("RunMat:fea:InvalidInput"),
    when: "A builtin receives an unsupported argument pattern or object type.",
    message: "fea: invalid input",
};

pub const ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.OPERATION_FAILED",
    identifier: Some("RunMat:fea:OperationFailed"),
    when: "A validation, planning, run, result-query, comparison, or trend operation fails.",
    message: "fea: operation failed",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INTERNAL",
    identifier: Some("RunMat:fea:Internal"),
    when: "An FEA object or operation result cannot be converted to a RunMat value.",
    message: "fea: internal error",
};

pub const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INPUT, ERROR_OPERATION, ERROR_INTERNAL];
