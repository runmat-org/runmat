//! Shared descriptor vocabulary for this runtime mechanism.
use crate::*;
pub const TEXT_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Function, script, file, command, or option text.",
}];

pub const DEBUG_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "A debugger compatibility helper receives more inputs than it supports.",
    message: "debugger helper: too many input arguments",
};

pub const DEBUG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.INVALID_INPUT",
    identifier: Some("RunMat:InvalidInput"),
    when: "An input is not a supported text, scalar, or option value.",
    message: "debugger helper: invalid input",
};

pub const DEBUG_ERROR_NO_CURRENT_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.NO_CURRENT_FILE",
    identifier: Some("RunMat:NoCurrentFile"),
    when: "`mlock`, `munlock`, or `mislocked` is called without a current function or script.",
    message: "debugger helper: no current function or script is available",
};

pub const DEBUG_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.FILE_NOT_FOUND",
    identifier: Some("RunMat:FileNotFound"),
    when: "`dbtype` cannot resolve a MATLAB source file.",
    message: "dbtype: file not found",
};

pub const DEBUG_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.FILE_READ",
    identifier: Some("RunMat:FileReadFailed"),
    when: "`dbtype` cannot read the resolved source file.",
    message: "dbtype: failed to read file",
};
