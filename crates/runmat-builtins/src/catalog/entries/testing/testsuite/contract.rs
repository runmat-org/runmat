use crate::*;

const TESTSUITE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tests",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("current folder"),
        description: "Test file, folder, function name, string array, or cell array of targets to discover without execution.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Common options such as IncludeSubfolders, BaseFolder, Name, ProcedureName, and UseParallel.",
    },
];

const TESTSUITE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "suite",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Scalar TestSuite object or homogeneous TestSuite object row; no test procedure is executed.",
}];

const TESTSUITE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "suite = testsuite",
        inputs: &[],
        outputs: &TESTSUITE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "suite = testsuite(tests, Name, Value, ...)",
        inputs: &TESTSUITE_INPUTS,
        outputs: &TESTSUITE_OUTPUT,
    },
];

pub const TESTSUITE_ERROR_REQUIRES_EXECUTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.REQUIRES_EXECUTOR",
    identifier: Some("RunMat:Testing:RequiresExecutor"),
    when: "`testsuite` is dispatched outside an active Core test executor.",
    message: "testsuite: requires an active Core test executor",
};

pub const TESTSUITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.INVALID_INPUT",
    identifier: Some("RunMat:testsuite:InvalidInput"),
    when: "A target or option value has an unsupported type or value.",
    message: "testsuite: invalid input",
};

pub const TESTSUITE_ERROR_UNSUPPORTED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.UNSUPPORTED_OPTION",
    identifier: Some("RunMat:testsuite:UnsupportedOption"),
    when: "A documented option requires an execution mode not available through the in-program adapter.",
    message: "testsuite: unsupported option",
};

pub const TESTSUITE_ERROR_TARGET_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.TARGET_NOT_FOUND",
    identifier: Some("RunMat:testsuite:TargetNotFound"),
    when: "A requested test target cannot be resolved to a file or folder.",
    message: "testsuite: test target not found",
};

pub const TESTSUITE_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.FILE_READ",
    identifier: Some("RunMat:testsuite:FileReadFailed"),
    when: "A discovered test file cannot be read as source text.",
    message: "testsuite: failed to read test file",
};

pub const TESTSUITE_ERROR_INSUFFICIENT_RESOURCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.INSUFFICIENT_RESOURCE",
    identifier: Some("RunMat:Placement:InsufficientResourceEvidence"),
    when: "Finite host-memory policy cannot certify discovery, frozen source, and result storage before source effects.",
    message: "testsuite: discovery, source, and result storage has no certified finite resource bound",
};

pub const TESTSUITE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTSUITE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:testsuite:TooManyOutputs"),
    when: "More than one result output is requested.",
    message: "testsuite: at most one output is supported",
};

pub const TESTSUITE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    TESTSUITE_ERROR_REQUIRES_EXECUTOR,
    TESTSUITE_ERROR_INVALID_INPUT,
    TESTSUITE_ERROR_UNSUPPORTED_OPTION,
    TESTSUITE_ERROR_TARGET_NOT_FOUND,
    TESTSUITE_ERROR_FILE_READ,
    TESTSUITE_ERROR_INSUFFICIENT_RESOURCE,
    TESTSUITE_ERROR_TOO_MANY_OUTPUTS,
];

pub const TESTSUITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TESTSUITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TESTSUITE_ERRORS,
};

const TESTSUITE_INTEGER_FLAG_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IncludeSubfolders, Coverage, or UseParallel",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric-or-logical flags accept only exact scalar zero or one; native integer storage is inspected without binary64 conversion.",
    }];
pub const TESTSUITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "suite = testsuite(..., flag_name, integer_0_or_1, ...)",
        inputs: &TESTSUITE_INTEGER_FLAG_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags control host semantic test discovery. Other direct integer arguments are invalid targets or option payloads, and resident numeric values reject before provider access.",
    }];
