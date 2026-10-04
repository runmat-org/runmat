use crate::*;

const COMMAND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "command",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host string scalar or character row containing a shell command.",
};
const ECHO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "echo",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "The literal '-echo' requests console output.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host process exit status as a double scalar.",
};
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "output",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complete stdout text followed by complete stderr text.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "status = system(command)",
        inputs: &[COMMAND],
        outputs: &[STATUS],
    },
    BuiltinSignatureDescriptor {
        label: "[status, output] = system(command)",
        inputs: &[COMMAND],
        outputs: &[STATUS, OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "[status, output] = system(command, '-echo')",
        inputs: &[COMMAND, ECHO],
        outputs: &[STATUS, OUTPUT],
    },
];

pub const SYSTEM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.INVALID_ARGUMENT",
    identifier: None,
    when: "The command or echo control is not host scalar text, the echo spelling is invalid, or the call requests more than two outputs.",
    message: "system: invalid command, option, or output count",
};
pub const SYSTEM_ERROR_UNSUPPORTED_PLATFORM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.UNSUPPORTED_PLATFORM",
    identifier: Some("RunMat:system:UnsupportedOnWasm"),
    when: "A valid process request is made in WebAssembly without a native process host.",
    message: "system: process execution is unavailable in WebAssembly",
};
pub const SYSTEM_ERROR_UNSUPPORTED_FILESYSTEM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.UNSUPPORTED_FILESYSTEM",
    identifier: Some("RunMat:system:UnsupportedFilesystem"),
    when: "The selected filesystem cannot issue a native process working-directory grant.",
    message: "system: selected filesystem cannot issue a native process directory",
};
pub const SYSTEM_ERROR_STALE_FILESYSTEM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.STALE_FILESYSTEM",
    identifier: Some("RunMat:system:StaleFilesystem"),
    when: "The selected process-directory grant revokes before output publication.",
    message: "system: selected process directory authority was revoked",
};
pub const SYSTEM_ERROR_MISSING_PROCESS_SERVICE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.MISSING_PROCESS_SERVICE",
    identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"),
    when: "An embedding invokes system without a native process service.",
    message: "system: no native process service is installed",
};
pub const SYSTEM_ERROR_INSUFFICIENT_RESOURCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.INSUFFICIENT_RESOURCE",
    identifier: Some("RunMat:Placement:InsufficientResourceEvidence"),
    when: "A finite host memory budget is configured but complete process output has no certified bound.",
    message: "system: complete child output has no certified finite resource bound",
};
pub const SYSTEM_ERROR_PROCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYSTEM.PROCESS",
    identifier: None,
    when: "The operating system cannot start, drain, or await the child process.",
    message: "system: child process failed",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    SYSTEM_ERROR_INVALID_ARGUMENT,
    SYSTEM_ERROR_UNSUPPORTED_PLATFORM,
    SYSTEM_ERROR_UNSUPPORTED_FILESYSTEM,
    SYSTEM_ERROR_STALE_FILESYSTEM,
    SYSTEM_ERROR_MISSING_PROCESS_SERVICE,
    SYSTEM_ERROR_INSUFFICIENT_RESOURCE,
    SYSTEM_ERROR_PROCESS,
];

pub const SYSTEM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const SYSTEM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Command and echo controls are host text; exact integer inputs reject before child creation. Exit status is a double output, not an integer input conversion.",
};
