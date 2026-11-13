use crate::*;

const ROOT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "rootkey",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host registry root selector.",
};
const KEY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "subkey",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host registry subkey path.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "valuename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Named value; empty text requests the default value.",
};
const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "'name'",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Literal 'name' requests value-name enumeration.",
};
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor { name: "value", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Character row for REG_SZ/REG_EXPAND_SZ, exact int32 for REG_DWORD, or cell column of value names." };
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "value = winqueryreg(rootkey, subkey)",
        inputs: &[ROOT, KEY],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "value = winqueryreg(rootkey, subkey, valuename)",
        inputs: &[ROOT, KEY, VALUE],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "names = winqueryreg('name', rootkey, subkey)",
        inputs: &[NAME, ROOT, KEY],
        outputs: &[OUTPUT],
    },
];
pub const WINQUERYREG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WINQUERYREG.INVALID_INPUT",
    identifier: Some("RunMat:winqueryreg:InvalidInput"),
    when: "The call does not provide two or three host scalar-text controls.",
    message:
        "winqueryreg: expected registry root, key, and optional value name as host scalar text",
};
pub const WINQUERYREG_ERROR_UNSUPPORTED_PLATFORM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WINQUERYREG.UNSUPPORTED_PLATFORM",
    identifier: Some("RunMat:winqueryreg:UnsupportedPlatform"),
    when: "Valid controls are supplied on a non-Windows target.",
    message: "winqueryreg: Windows registry is unavailable on this platform",
};
pub const WINQUERYREG_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WINQUERYREG.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:winqueryreg:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "winqueryreg: at most one output is supported",
};
pub const WINQUERYREG_ERROR_QUERY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WINQUERYREG.QUERY",
    identifier: Some("RunMat:winqueryreg:QueryFailed"),
    when: "reg.exe reports failure or returns no supported registry value.",
    message: "winqueryreg: registry query failed",
};
pub const WINQUERYREG_ERROR_INSUFFICIENT_RESOURCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.WINQUERYREG.INSUFFICIENT_RESOURCE", identifier: Some("RunMat:Placement:InsufficientResourceEvidence"), when: "A finite host-memory policy cannot certify complete process output before child creation.", message: "winqueryreg: complete child output has no certified finite resource bound" };
pub const WINQUERYREG_ERROR_PROCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.WINQUERYREG.PROCESS", identifier: Some("RunMat:winqueryreg:ProcessFailed"), when: "The selected native working directory, process service, child I/O, or wait operation fails.", message: "winqueryreg: native process failed" };
pub const WINQUERYREG_ERROR_STALE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WINQUERYREG.STALE",
    identifier: Some("RunMat:winqueryreg:StaleFilesystem"),
    when: "The selected native directory grant revokes before output publication.",
    message: "winqueryreg: process directory authority was revoked",
};
pub const WINQUERYREG_ERROR_CAPABILITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WINQUERYREG.CAPABILITY",
    identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"),
    when: "No native process service is installed in the selected RuntimeContext.",
    message: "winqueryreg: no native process service is installed",
};
pub const WINQUERYREG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        WINQUERYREG_ERROR_INVALID_INPUT,
        WINQUERYREG_ERROR_UNSUPPORTED_PLATFORM,
        WINQUERYREG_ERROR_TOO_MANY_OUTPUTS,
        WINQUERYREG_ERROR_QUERY,
        WINQUERYREG_ERROR_INSUFFICIENT_RESOURCE,
        WINQUERYREG_ERROR_PROCESS,
        WINQUERYREG_ERROR_STALE,
        WINQUERYREG_ERROR_CAPABILITY,
    ],
};

pub const WINQUERYREG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = winqueryreg(rootkey, subkey, valname) for REG_DWORD",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Text-only registry lookup inputs can return a documented exact int32 scalar for a 32-bit REG_DWORD value. The Windows command output is decoded as a 32-bit bit pattern without a floating conversion.",
    }];
