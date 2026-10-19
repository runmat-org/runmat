use crate::*;

const RUN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "script",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Script name or path to execute in the caller workspace.",
}];

const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "run(script)",
    inputs: &RUN_INPUTS,
    outputs: &[],
}];

pub const RUN_ERROR_REQUIRES_VM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.REQUIRES_VM",
    identifier: Some("RunMat:run:RequiresVm"),
    when: "`run` is dispatched without an active workspace execution context.",
    message: "run: requires an active workspace execution context",
};

pub const RUN_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.ARG_TYPE",
    identifier: Some("RunMat:run:InvalidScriptArgument"),
    when: "The script argument is not a character row, string scalar, or scalar string array.",
    message: "run: script must be a character vector or string scalar",
};

pub const RUN_ERROR_EMPTY_SCRIPT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.EMPTY_SCRIPT",
    identifier: Some("RunMat:run:EmptyScript"),
    when: "The script argument is an empty path.",
    message: "run: script path must not be empty",
};

pub const RUN_ERROR_PATH_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.PATH_RESOLVE",
    identifier: Some("RunMat:run:PathResolveFailed"),
    when: "RunMat cannot resolve the current directory, home directory, or search path.",
    message: "run: failed to resolve script path",
};

pub const RUN_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.FILE_NOT_FOUND",
    identifier: Some("RunMat:run:FileNotFound"),
    when: "No matching script file exists in the current directory or RunMat search path.",
    message: "run: script file not found",
};

pub const RUN_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.FILE_READ",
    identifier: Some("RunMat:run:FileReadFailed"),
    when: "The matched script file cannot be read as source text.",
    message: "run: failed to read script file",
};

pub const RUN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:run:TooManyOutputs"),
    when: "`run` is called with one or more requested output arguments.",
    message: "run: too many output arguments",
};

pub const RUN_ERRORS: [BuiltinErrorDescriptor; 7] = [
    RUN_ERROR_REQUIRES_VM,
    RUN_ERROR_ARG_TYPE,
    RUN_ERROR_EMPTY_SCRIPT,
    RUN_ERROR_PATH_RESOLVE,
    RUN_ERROR_FILE_NOT_FOUND,
    RUN_ERROR_FILE_READ,
    RUN_ERROR_TOO_MANY_OUTPUTS,
];

pub const RUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RUN_ERRORS,
};
