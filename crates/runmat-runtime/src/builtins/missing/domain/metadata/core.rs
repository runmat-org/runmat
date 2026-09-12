use super::super::*;

pub(in crate::builtins::missing::domain) const MISSING_TEXT: &str = "<missing>";

const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TF",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical missing-value mask.",
}];
const VALUE_AND_MASK_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Result value.",
    },
    BuiltinParamDescriptor {
        name: "TF",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Logical mask of entries, rows, or columns that were filled or removed.",
    },
];
const VALUE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
}];
const VARIADIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Size, method, dimension, or option arguments.",
}];
const VALUE_AND_ARGS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method, dimension, or option arguments.",
    },
];

const MISSING_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "missing",
        inputs: &[],
        outputs: &VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "missing(sz)",
        inputs: &VARIADIC_INPUTS,
        outputs: &VALUE_OUTPUT,
    },
];
const ONE_VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "TF = ismissing(A)",
    inputs: &VALUE_INPUT,
    outputs: &LOGICAL_OUTPUT,
}];
const ANYMISSING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "TF = anymissing(A)",
    inputs: &VALUE_INPUT,
    outputs: &LOGICAL_OUTPUT,
}];
pub(in crate::builtins::missing::domain) const FILLMISSING_SIGNATURES:
    [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = fillmissing(A, method, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_AND_MASK_OUTPUTS,
}];
pub(in crate::builtins::missing::domain) const RMMISSING_SIGNATURES: [BuiltinSignatureDescriptor;
    1] = [BuiltinSignatureDescriptor {
    label: "B = rmmissing(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_AND_MASK_OUTPUTS,
}];
pub(in crate::builtins::missing::domain) const STANDARDIZE_SIGNATURES:
    [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = standardizeMissing(A, indicators)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_OUTPUT,
}];
pub(in crate::builtins::missing::domain) const NANAWARE_SIGNATURES: [BuiltinSignatureDescriptor;
    1] = [BuiltinSignatureDescriptor {
    label: "B = nanmean(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_OUTPUT,
}];

const MISSING_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.INVALID_ARGUMENT",
    identifier: Some("RunMat:missing:InvalidArgument"),
    when: "Arguments do not match a supported missing-value syntax.",
    message: "missing-value builtin: invalid argument",
};
const MISSING_ERROR_UNSUPPORTED_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.UNSUPPORTED_TYPE",
    identifier: Some("RunMat:missing:UnsupportedType"),
    when: "The input type has no missing-value representation in RunMat.",
    message: "missing-value builtin: unsupported input type",
};
const MISSING_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.INTERNAL",
    identifier: Some("RunMat:missing:InternalError"),
    when: "Internal shape or table materialization fails.",
    message: "missing-value builtin: internal error",
};
pub(in crate::builtins::missing::domain) const MISSING_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MISSING_ERROR_INVALID_ARGUMENT,
    MISSING_ERROR_UNSUPPORTED_TYPE,
    MISSING_ERROR_INTERNAL,
];

pub(in crate::builtins::missing::domain) const MISSING_SHAPED_ARRAY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "missing-shaped-array",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "missing(size...) is a RunMat convenience; the documented MATLAB missing function accepts no input arguments",
    error_identifier: Some("RunMat:compatibility:MissingShapedArrayExtension"),
};
pub const MISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MISSING_SHAPED_ARRAY_EXTENSION];

pub(in crate::builtins::missing::domain) const STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "standardize-missing-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "standardizeMissing with a bare typed-integer input array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StandardizeMissingIntegerDataExtension"),
};
pub(in crate::builtins::missing::domain) const STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION: BuiltinExtensionDescriptor =
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

const STANDARDIZE_MISSING_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target's array-input datatype table excludes integer arrays. RunMat mode treats a bare integer array as an exact no-op because integer classes have no standard missing value.",
    }];
const STANDARDIZE_MISSING_INTEGER_INDICATOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "indicator",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly states that single, integer, and logical indicators also match double entries of A.",
    }];
const STANDARDIZE_MISSING_TABLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer table variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Table input is documented and preserves each variable datatype. Integer variables have no standard missing representation and therefore remain unchanged.",
    }];
pub const STANDARDIZE_MISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = standardizeMissing(integer_A, indicator)",
        inputs: &STANDARDIZE_MISSING_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The RunMat-only bare-array form preserves class, shape, and exact storage. Compatibility admission precedes provider access; automatic residency may gather transparently after admission.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = standardizeMissing(A, integer_indicator)",
        inputs: &STANDARDIZE_MISSING_INTEGER_INDICATOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer indicators are read from authoritative storage and compared in the documented target-class matching domain. Explicit gpuArray indicators are separately gated; automatic residency remains transparent.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = standardizeMissing(table_with_integer_variables, indicator)",
        inputs: &STANDARDIZE_MISSING_TABLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer table variables pass through with exact native storage while supported floating or textual variables are standardized independently.",
    },
];

const MISSING_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "size arguments",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every native integer size is read exactly and checked against nonnegative platform allocation limits; the entire shaped-array syntax is a RunMat-only convenience.",
    }];
pub const MISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "missing(integer_size, ...)", inputs: &MISSING_INTEGER_SIZE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "MATLAB-compatible modes reject every argument before provider access because public missing has only a zero-argument syntax. RunMat mode gathers admitted automatic or explicit size controls through their exact owner and creates a host string array." }];

descriptor!(
    MISSING_DESCRIPTOR,
    MISSING_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    ISMISSING_DESCRIPTOR,
    ONE_VALUE_SIGNATURES,
    BuiltinOutputMode::Fixed
);
pub(in crate::builtins::missing::domain) const ISMISSING_RESIDENT_INPUT_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ismissing-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ismissing with an interactive GPU-resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IsmissingResidentInputExtension"),
};
pub(in crate::builtins::missing) const ISMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ISMISSING_RESIDENT_INPUT_EXTENSION];
const ISMISSING_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight fixed-width integer classes have no standard missing value.",
    }];
pub const ISMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "TF = ismissing(integer_A)",
        inputs: &ISMISSING_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Returns a same-shaped all-false logical mask. Interactive resident input is a separately gated RunMat extension; admitted resident integers are validated from owner and class metadata without reading the payload and preserve this CPU builtin's host logical output policy.",
    }];
descriptor!(
    ANYMISSING_DESCRIPTOR,
    ANYMISSING_SIGNATURES,
    BuiltinOutputMode::Fixed
);

const ANYMISSING_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight built-in integer classes have no standard missing value.",
    }];
pub const ANYMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "TF = anymissing(integer_A)",
        inputs: &ANYMISSING_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer scalars and arrays return logical false because integer classes have no default missing representation; resident inputs gather without floating conversion.",
    }];

pub(in crate::builtins::missing::domain) fn logical_type(
    _args: &[Type],
    _ctx: &ResolveContext,
) -> Type {
    Type::Logical { shape: None }
}

pub(in crate::builtins::missing::domain) fn any_type(
    _args: &[Type],
    _ctx: &ResolveContext,
) -> Type {
    Type::Unknown
}

fn missing_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.into()))
        .with_builtin("missing");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(in crate::builtins::missing::domain) fn invalid_argument(
    detail: impl Into<String>,
) -> RuntimeError {
    missing_error(&MISSING_ERROR_INVALID_ARGUMENT, detail)
}

pub(in crate::builtins::missing::domain) fn unsupported_type(
    detail: impl Into<String>,
) -> RuntimeError {
    missing_error(&MISSING_ERROR_UNSUPPORTED_TYPE, detail)
}

pub(in crate::builtins::missing::domain) fn internal_error(
    detail: impl Into<String>,
) -> RuntimeError {
    missing_error(&MISSING_ERROR_INTERNAL, detail)
}
