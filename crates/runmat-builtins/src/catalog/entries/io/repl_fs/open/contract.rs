use crate::*;
const OPEN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MAT-file contents, extension-handler output, figure handle, or empty array.",
}];

const OPEN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File or variable name to open.",
}];

const OPEN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "open name",
        inputs: &OPEN_INPUTS,
        outputs: &OPEN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = open(name)",
        inputs: &OPEN_INPUTS,
        outputs: &OPEN_OUTPUT,
    },
];

pub const OPEN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.ARG_COUNT",
    identifier: Some("RunMat:open:ArgumentCount"),
    when: "The call does not provide exactly one input argument.",
    message: "open: expected exactly one input argument",
};

pub const OPEN_ERROR_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.NAME",
    identifier: Some("RunMat:open:InvalidName"),
    when: "The name is not a string scalar or row character vector.",
    message: "open: name must be a character vector or string scalar",
};

pub const OPEN_ERROR_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.NOT_FOUND",
    identifier: Some("RunMat:open:NotFound"),
    when: "No matching variable or file can be resolved.",
    message: "open: file or variable was not found",
};

pub const OPEN_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.UNSUPPORTED",
    identifier: Some("RunMat:open:UnsupportedFileType"),
    when: "The resolved file type cannot be opened by RunMat and no extension handler exists.",
    message: "open: unsupported file type",
};

pub const OPEN_ERROR_HANDLER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.HANDLER",
    identifier: Some("RunMat:open:HandlerFailed"),
    when: "A custom open<extension> handler fails.",
    message: "open: extension handler failed",
};

pub const OPEN_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.OUTPUT_COUNT",
    identifier: Some("RunMat:open:TooManyOutputs"),
    when: "The call requests more than one output.",
    message: "open: expected at most one output",
};

pub const OPEN_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.HOST",
    identifier: Some("RunMat:host:DocumentInteractionUnavailable"),
    when: "The selected editor or viewer route has no supporting application host.",
    message: "The active host does not provide document or editor interaction",
};
const OPEN_ERRORS: [BuiltinErrorDescriptor; 7] = [
    OPEN_ERROR_HOST,
    OPEN_ERROR_ARG_COUNT,
    OPEN_ERROR_NAME,
    OPEN_ERROR_NOT_FOUND,
    OPEN_ERROR_UNSUPPORTED,
    OPEN_ERROR_HANDLER,
    OPEN_ERROR_OUTPUT_COUNT,
];

pub const OPEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPEN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &OPEN_ERRORS,
};
