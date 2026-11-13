use crate::*;

const RUNTESTS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tests",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("current folder"),
        description: "Test file, folder, function name, string array, or cell array of targets.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Common options such as IncludeSubfolders, BaseFolder, Name, ProcedureName, and UseParallel.",
    },
];

const RUNTESTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Scalar TestResult object or homogeneous TestResult object row.",
}];

const RUNTESTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "results = runtests",
        inputs: &[],
        outputs: &RUNTESTS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "results = runtests(tests, Name, Value, ...)",
        inputs: &RUNTESTS_INPUTS,
        outputs: &RUNTESTS_OUTPUT,
    },
];

pub const RUNTESTS_ERROR_REQUIRES_EXECUTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.REQUIRES_EXECUTOR",
    identifier: Some("RunMat:Testing:RequiresExecutor"),
    when: "`runtests` is dispatched outside an active Core test executor.",
    message: "runtests: requires an active Core test executor",
};

pub const RUNTESTS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.INVALID_INPUT",
    identifier: Some("RunMat:runtests:InvalidInput"),
    when: "A target or option value has an unsupported type or value.",
    message: "runtests: invalid input",
};

pub const RUNTESTS_ERROR_UNSUPPORTED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.UNSUPPORTED_OPTION",
    identifier: Some("RunMat:runtests:UnsupportedOption"),
    when: "A documented option requires an execution mode not available through the in-program adapter.",
    message: "runtests: unsupported option",
};

pub const RUNTESTS_ERROR_TARGET_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.TARGET_NOT_FOUND",
    identifier: Some("RunMat:runtests:TargetNotFound"),
    when: "A requested test target cannot be resolved to a file or folder.",
    message: "runtests: test target not found",
};

pub const RUNTESTS_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.FILE_READ",
    identifier: Some("RunMat:runtests:FileReadFailed"),
    when: "A discovered test file cannot be read as source text.",
    message: "runtests: failed to read test file",
};

pub const RUNTESTS_ERROR_INSUFFICIENT_RESOURCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.INSUFFICIENT_RESOURCE",
    identifier: Some("RunMat:Placement:InsufficientResourceEvidence"),
    when: "Finite host-memory policy cannot certify discovery, frozen source, and result storage before source effects.",
    message: "runtests: discovery, source, and result storage has no certified finite resource bound",
};

pub const RUNTESTS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:runtests:TooManyOutputs"),
    when: "More than one result output is requested.",
    message: "runtests: at most one output is supported",
};

pub const RUNTESTS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    RUNTESTS_ERROR_REQUIRES_EXECUTOR,
    RUNTESTS_ERROR_INVALID_INPUT,
    RUNTESTS_ERROR_UNSUPPORTED_OPTION,
    RUNTESTS_ERROR_TARGET_NOT_FOUND,
    RUNTESTS_ERROR_FILE_READ,
    RUNTESTS_ERROR_INSUFFICIENT_RESOURCE,
    RUNTESTS_ERROR_TOO_MANY_OUTPUTS,
];

pub const RUNTESTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUNTESTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RUNTESTS_ERRORS,
};

const RUNTESTS_INTEGER_FLAG_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IncludeSubfolders, Coverage, or UseParallel",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric-or-logical flags accept only exact scalar zero or one; native integer storage is inspected without binary64 conversion.",
    }];
pub const RUNTESTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "results = runtests(..., flag_name, integer_0_or_1, ...)",
        inputs: &RUNTESTS_INTEGER_FLAG_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags control host test discovery and execution. Other direct integer arguments are invalid targets or option payloads, and resident numeric values reject before provider access.",
    }];
