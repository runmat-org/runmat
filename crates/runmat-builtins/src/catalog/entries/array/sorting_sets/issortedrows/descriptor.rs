use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when rows are sorted according to the requested row order.",
}];

const ISSORTEDROWS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input column vector, matrix, or table.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Column selectors, direction modes, ComparisonMethod, and MissingPlacement.",
    },
];

const ISSORTEDROWS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = issortedrows(A)",
        inputs: &ISSORTEDROWS_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issortedrows(A, column, direction, options)",
        inputs: &ISSORTEDROWS_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const ISSORTEDROWS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTEDROWS.INVALID_INPUT",
    identifier: Some("RunMat:issortedrows:InvalidInput"),
    when: "Input or row-sorting arguments are invalid.",
    message: "issortedrows: invalid input",
};

pub const ISSORTEDROWS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTEDROWS.INVALID_ARGUMENT",
    identifier: Some("RunMat:issortedrows:InvalidArgument"),
    when: "A column, direction, comparison, missing-placement, or output argument is invalid.",
    message: "issortedrows: invalid argument",
};

const ISSORTEDROWS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    ISSORTEDROWS_ERROR_INVALID_INPUT,
    ISSORTEDROWS_ERROR_INVALID_ARGUMENT,
];

pub const ISSORTEDROWS_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "issortedrows-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "issortedrows with an interactive resident GPU input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IssortedrowsGpuInputExtension"),
    };

pub const ISSORTEDROWS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ISSORTEDROWS_GPU_INPUT_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented matrix domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "column",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonzero integer column scalar or vector accepts every integer class and integer-valued double.",
    },
];

pub const ISSORTEDROWS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = issortedrows(integer_A, integer_columns, direction, options)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer rows and column selectors are compared exactly across ascending, descending, monotonic, strict, and per-column direction forms. tf is scalar logical. Interactive GPU input is a mode-gated RunMat extension that gathers exact typed storage.",
    }];

pub const ISSORTEDROWS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSORTEDROWS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSORTEDROWS_ERRORS,
};
