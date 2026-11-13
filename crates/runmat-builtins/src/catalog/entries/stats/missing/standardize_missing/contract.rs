use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, text, cell, or table data.",
    },
    BuiltinParamDescriptor {
        name: "indicators",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One or more numeric or text sentinel values.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Data with matching values changed to the class-appropriate missing representation.",
}];
pub const STANDARDIZE_MISSING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "B = standardizeMissing(A, indicators)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INVALID_ARGUMENT",
            identifier: Some("RunMat:missing:InvalidArgument"),
            when: "Required arguments are absent or a resident input cannot be admitted.",
            message: "missing-value builtin: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.UNSUPPORTED_TYPE",
            identifier: Some("RunMat:missing:UnsupportedType"),
            when: "An indicator or input form has no supported missing-value conversion.",
            message: "missing-value builtin: unsupported input type",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INTERNAL",
            identifier: Some("RunMat:missing:InternalError"),
            when: "Internal array or table materialization fails.",
            message: "missing-value builtin: internal error",
        },
    ],
};

pub const STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "standardize-missing-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "standardizeMissing with a bare typed-integer input array is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StandardizeMissingIntegerDataExtension"),
    };
pub const STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "standardize-missing-explicit-gpu-indicator",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "standardizeMissing with an explicitly GPU-resident indicator is a RunMat extension",
        error_identifier: Some(
            "RunMat:compatibility:StandardizeMissingExplicitGpuIndicatorExtension",
        ),
    };
pub const STANDARDIZE_MISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION,
    STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION,
];

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Direct native integer data has no standard missing value and is an exact no-op in RunMat mode.",
}];
const INTEGER_INDICATORS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "indicators",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer indicators compare to floating entries without rounding the indicator. An inexact wide integer cannot falsely match a floating value.",
    }];
const TABLE_INTEGERS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "integer table variables",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer table variables retain their native class and values.",
}];
pub const STANDARDIZE_MISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = standardizeMissing(integer_A, indicators)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only direct-integer form preserves exact storage. Compatibility admission precedes provider access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = standardizeMissing(A, integer_indicators)",
        inputs: &INTEGER_INDICATORS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Indicators are read from authoritative integer storage and matched only when the floating value represents the same integer exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = standardizeMissing(table_with_integer_variables, indicators)",
        inputs: &TABLE_INTEGERS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Table variables are processed independently; integer variables remain exact while supported floating and text variables change.",
    },
];
