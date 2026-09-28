use crate::*;

pub const REGEXPREP_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "regexprep-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "regexprep host fallback for explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RegexprepExplicitGpuInputExtension"),
    };
pub const REGEXPREP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [REGEXPREP_EXPLICIT_GPU_EXTENSION];

const REGEXPREP_INTEGER_OCCURRENCE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "N",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented Nth-occurrence option accepts every integer class and is validated exactly as a positive scalar before regex execution.",
    }];
const REGEXPREP_INTEGER_TEXT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "str/expression/replace",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Regex text roles reject integer host and resident values before gather or numeric-to-text conversion.",
    }];
pub const REGEXPREP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "out = regexprep(subject, pattern, replacement, integer_N)",
        inputs: &REGEXPREP_INTEGER_OCCURRENCE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "N selects exactly one eligible match without entering the floating domain. Automatic residency gathers transparently; explicit gpuArray fallback is separately gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "out = regexprep(integer_subject_or_pattern_or_replacement, ...)",
        inputs: &REGEXPREP_INTEGER_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer data is not text and never triggers provider access or implicit decimal formatting in these roles.",
    },
];

const REGEXPREP_OUTPUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with replacements applied (preserves scalar/array shape semantics).",
}];

const REGEXPREP_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "subject",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (char/string/string-array/cellstr).",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (scalar, sequence, or elementwise list).",
    },
    BuiltinParamDescriptor {
        name: "replacement",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replacement text (scalar, sequence, or elementwise list).",
    },
];

const REGEXPREP_INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "subject",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (char/string/string-array/cellstr).",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (scalar, sequence, or elementwise list).",
    },
    BuiltinParamDescriptor {
        name: "replacement",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replacement text (scalar, sequence, or elementwise list).",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Regex replacement options (case/lineanchors/dotall/once/emptymatch/...).",
    },
];

const REGEXPREP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "out = regexprep(subject, pattern, replacement)",
        inputs: &REGEXPREP_INPUTS_CORE,
        outputs: &REGEXPREP_OUTPUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "out = regexprep(subject, pattern, replacement, options...)",
        inputs: &REGEXPREP_INPUTS_OPTIONS,
        outputs: &REGEXPREP_OUTPUT_TEXT,
    },
];

const REGEXPREP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXPREP.INVALID_ARGUMENT",
    identifier: Some("RunMat:regexprep:InvalidArgument"),
    when: "Input/options are malformed, unsupported, or shape-incompatible.",
    message: "regexprep: invalid argument",
};

const REGEXPREP_ERROR_PATTERN_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXPREP.PATTERN_INVALID",
    identifier: Some("RunMat:regexprep:PatternInvalid"),
    when: "Pattern cannot be compiled as a regular expression.",
    message: "regexprep: invalid regular expression pattern",
};

const REGEXPREP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXPREP.INTERNAL",
    identifier: Some("RunMat:regexprep:Internal"),
    when: "Internal regex replacement or output reconstruction fails.",
    message: "regexprep: internal operation failed",
};

const REGEXPREP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    REGEXPREP_ERROR_INVALID_ARGUMENT,
    REGEXPREP_ERROR_PATTERN_INVALID,
    REGEXPREP_ERROR_INTERNAL,
];

pub const REGEXPREP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGEXPREP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REGEXPREP_ERRORS,
};
