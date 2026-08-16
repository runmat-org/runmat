//! MATLAB-compatible missing-value construction, predicates, cleanup, and NaN-aware aliases.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, IntValue, LogicalArray, NumericDType, ObjectInstance, ResolveContext,
    StringArray, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::builtins::math::reduction::{mean, median, min, std as std_reduction, sum, var};
use crate::builtins::table::{
    is_tabular_object, select_rows, selected_row_names, table_from_columns_like, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MISSING_TEXT: &str = "<missing>";

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
const FILLMISSING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = fillmissing(A, method, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_AND_MASK_OUTPUTS,
}];
const RMMISSING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = rmmissing(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_AND_MASK_OUTPUTS,
}];
const STANDARDIZE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = standardizeMissing(A, indicators)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &VALUE_OUTPUT,
}];
const NANAWARE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
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
const MISSING_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MISSING_ERROR_INVALID_ARGUMENT,
    MISSING_ERROR_UNSUPPORTED_TYPE,
    MISSING_ERROR_INTERNAL,
];

const MISSING_SHAPED_ARRAY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "missing-shaped-array",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "missing(size...) is a RunMat convenience; the documented MATLAB missing function accepts no input arguments",
    error_identifier: Some("RunMat:compatibility:MissingShapedArrayExtension"),
};
pub const MISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MISSING_SHAPED_ARRAY_EXTENSION];

const STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "standardize-missing-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "standardizeMissing with a bare typed-integer input array is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StandardizeMissingIntegerDataExtension"),
    };
const STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION: BuiltinExtensionDescriptor =
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

macro_rules! descriptor {
    ($name:ident, $signatures:ident, $mode:expr) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: $mode,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &MISSING_ERRORS,
        };
    };
}

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
const ISMISSING_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ismissing-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ismissing with an interactive GPU-resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IsmissingResidentInputExtension"),
};
const ISMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ISMISSING_RESIDENT_INPUT_EXTENSION];
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

const RMMISSING_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rmmissing-integer-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rmmissing accepts a typed-integer dimension control as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RmmissingIntegerDimensionExtension"),
};
pub const RMMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RMMISSING_INTEGER_DIM_EXTENSION];
const RMMISSING_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The R2022a-and-later surface accepts datatypes without a standard missing definition; all eight integer classes therefore remain unchanged.",
    }];
const RMMISSING_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat additionally accepts a typed integer dimension scalar and reads it exactly as a structural control.",
    }];
pub const RMMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[R,TF] = rmmissing(integer_A, ...)",
        inputs: &RMMISSING_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer A is an exact no-op because it has no standard missing value; R preserves class and storage, TF is all-false logical, and documented resident outputs return through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rmmissing(A, integer_dim)",
        inputs: &RMMISSING_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The dimension extension is mode-gated before provider access and never crosses a floating boundary.",
    },
];

const FILLMISSING_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fillmissing-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fillmissing with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FillmissingIntegerDataExtension"),
};
const FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fillmissing-aggregate-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fillmissing with integer data nested in a table or cell array is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension"),
    };
pub const FILLMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    FILLMISSING_INTEGER_DATA_EXTENSION,
    FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION,
];
const FILLMISSING_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer arrays have no standard missing value. RunMat mode preserves authoritative same-class storage and returns an all-false filled-entry mask.",
    }];
const FILLMISSING_AGGREGATE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "table variables or nested cell contents",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The aggregate is recursively classified before any resident child can be gathered; integer children retain exact same-class storage and contribute false entries to the filled mask.",
    }];
pub const FILLMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[F, TF] = fillmissing(integer_A, method, ...)",
        inputs: &FILLMISSING_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The RunMat-only integer form is an exact no-op because integer classes have no default missing representation; TF is logical false with the input shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[F, TF] = fillmissing(table_or_cell_with_integer_data, method, ...)",
        inputs: &FILLMISSING_AGGREGATE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Nested table/cell integer data is a separately declared RunMat-only aggregate extension and is classified recursively before provider access.",
    },
];
descriptor!(
    FILLMISSING_DESCRIPTOR,
    FILLMISSING_SIGNATURES,
    BuiltinOutputMode::ByRequestedOutputCount
);
descriptor!(
    RMMISSING_DESCRIPTOR,
    RMMISSING_SIGNATURES,
    BuiltinOutputMode::ByRequestedOutputCount
);
descriptor!(
    STANDARDIZE_MISSING_DESCRIPTOR,
    STANDARDIZE_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    NAN_AWARE_DESCRIPTOR,
    NANAWARE_SIGNATURES,
    BuiltinOutputMode::Fixed
);

macro_rules! integer_extension {
    ($descriptor:ident, $extensions:ident, $id:literal, $description:literal, $error:literal) => {
        const $descriptor: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
        pub const $extensions: [BuiltinExtensionDescriptor; 1] = [$descriptor];
    };
}

integer_extension!(
    NANMEAN_INTEGER_EXTENSION,
    NANMEAN_EXTENSIONS,
    "nanmean-typed-integer-input",
    "nanmean with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NanmeanTypedIntegerInputExtension"
);
integer_extension!(
    NANSUM_INTEGER_EXTENSION,
    NANSUM_EXTENSIONS,
    "nansum-typed-integer-input",
    "nansum with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NansumTypedIntegerInputExtension"
);
integer_extension!(
    NANMIN_INTEGER_EXTENSION,
    NANMIN_EXTENSIONS,
    "nanmin-typed-integer-input",
    "nanmin with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NanminTypedIntegerInputExtension"
);
integer_extension!(
    NANMEDIAN_INTEGER_EXTENSION,
    NANMEDIAN_EXTENSIONS,
    "nanmedian-typed-integer-input",
    "nanmedian with a typed-integer data or control input is a RunMat extension",
    "RunMat:compatibility:NanmedianTypedIntegerInputExtension"
);
integer_extension!(
    NANSTD_INTEGER_CONTROL_EXTENSION,
    NANSTD_EXTENSIONS,
    "nanstd-typed-integer-control",
    "nanstd with a typed-integer normalization or dimension input is a RunMat extension",
    "RunMat:compatibility:NanstdTypedIntegerControlExtension"
);
integer_extension!(
    NANVAR_INTEGER_CONTROL_EXTENSION,
    NANVAR_EXTENSIONS,
    "nanvar-typed-integer-control",
    "nanvar with a typed-integer normalization or dimension input is a RunMat extension",
    "RunMat:compatibility:NanvarTypedIntegerControlExtension"
);

const MOVMAD_GPU_LARGE_WINDOW_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "movmad-gpu-large-window",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "movmad with a window longer than 31 on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MovmadGpuLargeWindowExtension"),
};

pub const MOVMAD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MOVMAD_GPU_LARGE_WINDOW_EXTENSION];

const MOVMAD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Moving median absolute deviation accepts every real integer input class and returns double deviation values.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Count windows accept exact positive typed-integer lengths or integer-valued floating lengths.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional positive scalar dimension accepts exact typed integers or integer-valued floating values.",
    },
];

pub const MOVMAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "M = movmad(A, k, dim, nanflag)",
        inputs: &MOVMAD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "On the currently supported scalar-window surface, integer observations materialize once into the double median-absolute-deviation domain; supported resident inputs gather and re-upload double output, while GPU windows longer than 31 are separately mode-gated.",
    }];

const RUNMAT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_or_control",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer data, pairwise operands, and dimension controls are accepted only with the builtin's declared RunMat extension; documented single- and double-valued forms remain available in MATLAB-compatible mode.",
    }];

const REJECTED_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed-integer data is rejected before host or provider reduction.",
}];

const RUNMAT_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "normalization_or_dimension",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer normalization or dimension controls are accepted only with the builtin's declared RunMat extension; integer-valued double controls remain documented-compatible.",
    }];

pub const NANMEAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmean(A, args...)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmean by routing typed-integer forms through mean(...,\"omitnan\"); default/double output is double, native output preserves the input class, and modern mean-only options remain extension syntax.",
    }];

pub const NANSUM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nansum(A, args...)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nansum by routing typed-integer forms through sum(...,\"omitnan\"); default/double output is double, native output preserves the input class with saturating accumulation, and modern sum-only options remain extension syntax.",
    }];

pub const NANMIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmin(A, args...) or nanmin(A, B)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmin with exact typed-integer reduction and compatible pairwise forms; omit-NaN resident reductions use host fallback, while pairwise execution follows min's same-class-or-scalar-double rules.",
    }];

pub const NANMEDIAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "y = nanmedian(A, args...)",
        inputs: &RUNMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat extends legacy nanmedian by routing typed-integer forms through median(...,\"omitnan\"); exact host reduction preserves all eight classes and resident fallback output is re-uploaded.",
    }];

pub const NANSTD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanstd(A, args...) with typed-integer A",
        inputs: &REJECTED_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer data is unsupported in both compatibility modes and is rejected before provider dispatch.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanstd(A, flag_or_dimension)",
        inputs: &RUNMAT_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "With floating data, RunMat accepts exact typed-integer normalization and dimension controls only when the nanstd typed-integer-control extension is enabled.",
    },
];

pub const NANVAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanvar(A, args...) with typed-integer A",
        inputs: &REJECTED_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer data is unsupported in both compatibility modes and is rejected before provider dispatch.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = nanvar(A, normalization_or_dimension)",
        inputs: &RUNMAT_INTEGER_CONTROLS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "With floating data, RunMat accepts exact typed-integer normalization and dimension controls only when the nanvar typed-integer-control extension is enabled; non-scalar weighted variance remains separately unsupported.",
    },
];

fn logical_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Logical { shape: None }
}

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
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

fn invalid_argument(detail: impl Into<String>) -> RuntimeError {
    missing_error(&MISSING_ERROR_INVALID_ARGUMENT, detail)
}

fn unsupported_type(detail: impl Into<String>) -> RuntimeError {
    missing_error(&MISSING_ERROR_UNSUPPORTED_TYPE, detail)
}

fn internal_error(detail: impl Into<String>) -> RuntimeError {
    missing_error(&MISSING_ERROR_INTERNAL, detail)
}

#[runtime_builtin(
    name = "missing",
    category = "missing",
    summary = "Create MATLAB missing string scalars or arrays.",
    keywords = "missing,string,missing values",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::MISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::MISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::MISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn missing_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MISSING_SHAPED_ARRAY_EXTENSION,
            "missing",
        )?;
    }
    let packed = Value::OutputList(args);
    let gathered = gather_if_needed_async(&packed)
        .await
        .map_err(|err| invalid_argument(format!("missing: failed to gather arguments: {err}")))?;
    let args = match gathered {
        Value::OutputList(values) => values,
        _ => Vec::new(),
    };
    let shape = parse_size_args(&args)?;
    missing_string_array(shape)
}

#[runtime_builtin(
    name = "ismissing",
    category = "missing",
    summary = "Return a logical mask identifying missing values.",
    keywords = "ismissing,missing,NaN,NaT,string,table",
    accel = "cpu",
    type_resolver(logical_type),
    descriptor(crate::builtins::missing::ISMISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::ISMISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::ISMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn ismissing_builtin(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = &value {
        if runmat_accelerate_api::handle_is_explicit(handle) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ISMISSING_RESIDENT_INPUT_EXTENSION,
                "ismissing",
            )?;
        }
        if runmat_accelerate_api::handle_integer_type(handle).is_some() {
            return ismissing_resident_integer(handle);
        }
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("ismissing: failed to gather input: {err}")))?;
    ismissing_value(&value)
}

fn ismissing_resident_integer(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let integer = runmat_accelerate_api::handle_integer_type(handle)
        .expect("resident integer predicate requires integer metadata");
    let storage = runmat_accelerate_api::handle_storage(handle);
    if gpu_helpers::exact_provider_for_handle(handle).is_none()
        || storage != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
    {
        return Err(internal_error(
            "ismissing: resident integer metadata is contradictory",
        ));
    }
    Ok(Value::LogicalArray(LogicalArray::zeros(
        handle.shape.clone(),
    )))
}

#[runtime_builtin(
    name = "anymissing",
    category = "missing",
    summary = "Return true when an input contains at least one missing value.",
    keywords = "anymissing,missing,NaN,string,table",
    accel = "cpu",
    type_resolver(logical_type),
    descriptor(crate::builtins::missing::ANYMISSING_DESCRIPTOR),
    integer_capabilities(crate::builtins::missing::ANYMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn anymissing_builtin(value: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("anymissing: failed to gather input: {err}")))?;
    Ok(Value::Bool(any_missing(&value)?))
}

#[runtime_builtin(
    name = "standardizeMissing",
    category = "missing",
    summary = "Replace user-specified missing indicators with canonical missing values.",
    keywords = "standardizeMissing,missing,NaN,string,table",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::STANDARDIZE_MISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::STANDARDIZE_MISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::STANDARDIZE_MISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn standardize_missing_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION,
            "standardizeMissing",
        )?;
    }
    let indicator = rest
        .first()
        .ok_or_else(|| invalid_argument("standardizeMissing: missing indicators argument"))?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(indicator) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION,
            "standardizeMissing",
        )?;
    }
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        invalid_argument(format!("standardizeMissing: failed to gather input: {err}"))
    })?;
    let indicator = gather_if_needed_async(indicator).await.map_err(|err| {
        invalid_argument(format!(
            "standardizeMissing: failed to gather indicators: {err}"
        ))
    })?;
    let indicators = indicator_set(&indicator)?;
    standardize_missing_value(value, &indicators)
}

#[runtime_builtin(
    name = "rmmissing",
    category = "missing",
    summary = "Remove missing elements, rows, or columns.",
    keywords = "rmmissing,missing,NaN,string,table",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::RMMISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::RMMISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::RMMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn rmmissing_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMMISSING_INTEGER_DIM_EXTENSION,
            "rmmissing",
        )?;
    }
    let source = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let value = if let Some(handle) = source.as_ref() {
        let owner = gpu_helpers::exact_provider_for_handle(handle)
            .ok_or_else(|| invalid_argument("rmmissing: no provider owns the resident input"))?;
        gpu_helpers::download_value_preserving_residency_async(owner, handle)
            .await
            .map_err(|err| invalid_argument(format!("rmmissing: failed to gather input: {err}")))?
    } else {
        value
    };
    let options = RemoveOptions::parse(&rest)?;
    let (result, removed) = remove_missing_value(value, options)?;
    let (result, removed) = if let Some(source) = source.as_ref() {
        (
            gpu_helpers::restore_class_preserving_value(source, result, "rmmissing")?,
            gpu_helpers::restore_class_preserving_value(
                source,
                Value::LogicalArray(removed),
                "rmmissing",
            )?,
        )
    } else {
        (result, Value::LogicalArray(removed))
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![result, removed],
        )),
        None => Ok(result),
    }
}

#[runtime_builtin(
    name = "fillmissing",
    category = "missing",
    summary = "Fill missing entries using constant, neighbor, or summary methods.",
    keywords = "fillmissing,missing,NaN,string,table,previous,next,linear,constant",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::FILLMISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::FILLMISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::FILLMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn fillmissing_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if is_real_typed_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLMISSING_INTEGER_DATA_EXTENSION,
            "fillmissing",
        )?;
    } else if fillmissing_aggregate_contains_integer(&value)? {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION,
            "fillmissing",
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("fillmissing: failed to gather input: {err}")))?;
    let options = FillOptions::parse(&rest)?;
    let (result, mask) = fill_missing_value(value, &options)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![result, Value::LogicalArray(mask)],
        )),
        None => Ok(result),
    }
}

fn fillmissing_aggregate_contains_integer(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Cell(cell) => {
            for child in &cell.data {
                if is_real_typed_integer_value(child)
                    || fillmissing_aggregate_contains_integer(child)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Value::Object(object) if is_tabular_object(object) => {
            let variables = table_variables(object)?;
            for child in variables.fields.values() {
                if is_real_typed_integer_value(child)
                    || fillmissing_aggregate_contains_integer(child)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

#[runtime_builtin(
    name = "nanmean",
    category = "missing",
    summary = "Mean that ignores NaN values.",
    keywords = "nanmean,mean,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANMEAN_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANMEAN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn nanmean_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANMEAN_INTEGER_EXTENSION, "nanmean", &value, &rest)?;
    mean::mean_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nansum",
    category = "missing",
    summary = "Sum that ignores NaN values.",
    keywords = "nansum,sum,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANSUM_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANSUM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn nansum_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANSUM_INTEGER_EXTENSION, "nansum", &value, &rest)?;
    sum::sum_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanmin",
    category = "missing",
    summary = "Minimum that ignores NaN values.",
    keywords = "nanmin,min,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANMIN_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANMIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn nanmin_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANMIN_INTEGER_EXTENSION, "nanmin", &value, &rest)?;
    if let Some(first) = rest.first() {
        if is_numeric_data_like(first) {
            if rest.len() != 1 {
                return Err(invalid_argument(
                    "nanmin: pairwise form accepts exactly two numeric inputs",
                ));
            }
            return pairwise_nan_min(value, first.clone());
        }
    }
    min::min_builtin(value, nanmin_rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanmedian",
    category = "missing",
    summary = "Median that ignores NaN values.",
    keywords = "nanmedian,median,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANMEDIAN_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANMEDIAN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn nanmedian_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANMEDIAN_INTEGER_EXTENSION, "nanmedian", &value, &rest)?;
    median::median_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanstd",
    category = "missing",
    summary = "Standard deviation that ignores NaN values.",
    keywords = "nanstd,std,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANSTD_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANSTD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn nanstd_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_control_extension(
        &NANSTD_INTEGER_CONTROL_EXTENSION,
        "nanstd",
        &value,
        &rest,
    )?;
    std_reduction::std_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanvar",
    category = "missing",
    summary = "Variance that ignores NaN values.",
    keywords = "nanvar,var,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANVAR_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANVAR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn nanvar_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_control_extension(
        &NANVAR_INTEGER_CONTROL_EXTENSION,
        "nanvar",
        &value,
        &rest,
    )?;
    var::var_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "movmad",
    category = "missing",
    summary = "Moving median absolute deviation over vectors and matrix dimensions.",
    keywords = "movmad,moving,median,absolute,deviation,missing",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::MOVMAD_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::MOVMAD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
async fn movmad_builtin(value: Value, window: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let window = scalar_usize(&window, "movmad window")?;
    let provider = match &value {
        Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle)
            .or_else(runmat_accelerate_api::provider),
        _ => None,
    };
    if provider.is_some() && window > 31 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MOVMAD_GPU_LARGE_WINDOW_EXTENSION,
            "movmad",
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("movmad: failed to gather input: {err}")))?;
    let tensor = numeric_tensor(value, "movmad")?;
    let options = MovingOptions::parse(&rest)?;
    let result = moving_mad(tensor, window, options)?;
    match (provider, result) {
        (Some(provider), Value::Tensor(tensor)) => {
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|err| internal_error(format!("movmad: failed to upload result: {err}")))?;
            Ok(Value::GpuTensor(handle))
        }
        (_, result) => Ok(result),
    }
}

fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn ensure_nan_integer_extension(
    extension: &BuiltinExtensionDescriptor,
    builtin: &str,
    value: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    if is_real_typed_integer_value(value) || rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

fn ensure_nan_integer_control_extension(
    extension: &BuiltinExtensionDescriptor,
    builtin: &str,
    value: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    if !is_real_typed_integer_value(value) && rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

fn rest_with_omitnan(mut rest: Vec<Value>) -> Vec<Value> {
    let insert_at = rest
        .iter()
        .position(|arg| scalar_text(arg).is_some_and(|text| text.eq_ignore_ascii_case("like")))
        .unwrap_or(rest.len());
    rest.insert(insert_at, Value::from("omitnan"));
    rest
}

fn nanmin_rest_with_omitnan(mut rest: Vec<Value>) -> Vec<Value> {
    if rest.is_empty() {
        rest.push(Value::Tensor(
            Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty placeholder shape"),
        ));
    }
    rest.push(Value::from("omitnan"));
    rest
}

fn parse_size_args(args: &[Value]) -> BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        return Ok(vec![1, 1]);
    }
    if args.len() == 1 {
        match &args[0] {
            Value::Tensor(tensor) => return tensor_shape_as_size(tensor),
            Value::Int(_) | Value::Num(_) => {
                let n = scalar_usize(&args[0], "missing size")?;
                return Ok(vec![n, n]);
            }
            Value::String(s) if s.eq_ignore_ascii_case("like") => {
                return Err(invalid_argument(
                    "missing: 'like' requires a prototype value",
                ));
            }
            _ => {}
        }
    }
    let mut out = Vec::with_capacity(args.len());
    let mut idx = 0;
    while idx < args.len() {
        if scalar_text(&args[idx])
            .map(|text| text.eq_ignore_ascii_case("like"))
            .unwrap_or(false)
        {
            idx += 2;
            continue;
        }
        out.push(scalar_usize(&args[idx], "missing size")?);
        idx += 1;
    }
    if out.is_empty() {
        Ok(vec![1, 1])
    } else {
        Ok(out)
    }
}

fn tensor_shape_as_size(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    if let Some(storage) = tensor.integer_storage() {
        return (0..storage.len())
            .map(|index| {
                let value = storage.value_at(index).ok_or_else(|| {
                    internal_error("missing: integer size vector storage length mismatch")
                })?;
                integer_size_to_usize(&value, "missing size")
            })
            .collect();
    }
    let values = tensor_utils::tensor_values_f64_cow(tensor);
    if values.is_empty() {
        return Ok(vec![0, 0]);
    }
    values
        .iter()
        .map(|value| {
            if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
                return Err(invalid_argument(
                    "missing: sizes must be nonnegative finite integers",
                ));
            }
            if *value > usize::MAX as f64 {
                return Err(invalid_argument("missing: size exceeds platform limits"));
            }
            if usize::BITS == 64 && *value == usize::MAX as f64 {
                return Err(invalid_argument("missing: size exceeds platform limits"));
            }
            Ok(*value as usize)
        })
        .collect()
}

fn missing_string_array(shape: Vec<usize>) -> BuiltinResult<Value> {
    let count = element_count(&shape)?;
    let array =
        StringArray::new(vec![MISSING_TEXT.to_string(); count], shape).map_err(internal_error)?;
    Ok(Value::StringArray(array))
}

fn element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| invalid_argument("array size is too large"))
    })
}

fn ismissing_value(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Bool(n.is_nan())),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_nan() || im.is_nan())),
        Value::Int(_) | Value::Bool(_) | Value::FunctionHandle(_) | Value::ClassRef(_) => {
            Ok(Value::Bool(false))
        }
        Value::String(s) => Ok(Value::Bool(is_missing_text(s))),
        Value::CharArray(array) => Ok(Value::LogicalArray(
            LogicalArray::new(
                char_rows(array)
                    .into_iter()
                    .map(|text| u8::from(text.trim().is_empty() || is_missing_text(&text)))
                    .collect(),
                vec![array.rows, 1],
            )
            .map_err(internal_error)?,
        )),
        Value::StringArray(array) => logical_from_iter(
            array.data.iter().map(|text| is_missing_text(text)),
            array.shape.clone(),
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => logical_from_iter(
            vec![false; tensor_utils::tensor_element_len(tensor)],
            tensor.shape.clone(),
        ),
        Value::Tensor(tensor) => {
            let values = tensor_utils::tensor_values_f64_cow(tensor);
            logical_from_iter(
                values.iter().map(|value| value.is_nan()),
                tensor.shape.clone(),
            )
        }
        Value::ComplexTensor(tensor) => logical_from_iter(
            tensor
                .materialize_f64()
                .iter()
                .map(|(re, im)| re.is_nan() || im.is_nan()),
            tensor.shape.clone(),
        ),
        Value::SparseTensor(tensor) => {
            let mut data = vec![0u8; tensor.rows * tensor.cols];
            if tensor.integer_storage().is_none() {
                for col in 0..tensor.cols {
                    for idx in tensor.col_ptrs[col]..tensor.col_ptrs[col + 1] {
                        if tensor
                            .numeric_value_at(idx)
                            .expect("sparse storage index is valid")
                            .materialize_f64()
                            .is_nan()
                        {
                            data[tensor.row_indices[idx] + col * tensor.rows] = 1;
                        }
                    }
                }
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![tensor.rows, tensor.cols]).map_err(internal_error)?,
            ))
        }
        Value::LogicalArray(array) => Ok(Value::LogicalArray(LogicalArray::zeros(
            array.shape.clone(),
        ))),
        Value::Cell(cell) => {
            let mut data = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                data.push(u8::from(any_missing(item)?));
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![cell.rows, cell.cols]).map_err(internal_error)?,
            ))
        }
        Value::Struct(st) => {
            let mut out = StructValue::new();
            for (name, field) in &st.fields {
                out.insert(name.clone(), ismissing_value(field)?);
            }
            Ok(Value::Struct(out))
        }
        Value::Object(object) if is_tabular_object(object) => ismissing_table(object),
        Value::Object(object) if object.is_class("datetime") => {
            let serials = crate::builtins::datetime::serials_from_datetime_value(value)?;
            let values = tensor_utils::tensor_values_f64_cow(&serials);
            logical_from_iter(
                values.iter().map(|serial| serial.is_nan()),
                serials.shape.clone(),
            )
        }
        Value::Object(object) if object.is_class("duration") => {
            let days = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let values = tensor_utils::tensor_values_f64_cow(&days);
            logical_from_iter(values.iter().map(|day| day.is_nan()), days.shape.clone())
        }
        Value::OutputList(values) => {
            let mut data = Vec::with_capacity(values.len());
            for item in values {
                data.push(u8::from(any_missing(item)?));
            }
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![1, values.len()]).map_err(internal_error)?,
            ))
        }
        _ => Ok(Value::Bool(false)),
    }
}

fn ismissing_table(object: &ObjectInstance) -> BuiltinResult<Value> {
    let height = table_height(object)?;
    let width = table_width(object)?;
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let mut data = vec![0u8; height * width];
    for (col, name) in names.iter().enumerate() {
        let Some(value) = variables.fields.get(name) else {
            continue;
        };
        let mask = logical_mask_for_rows(value, height)?;
        for row in 0..height {
            if mask.get(row).copied().unwrap_or(0) != 0 {
                data[row + col * height] = 1;
            }
        }
    }
    Ok(Value::LogicalArray(
        LogicalArray::new(data, vec![height, width]).map_err(internal_error)?,
    ))
}

fn logical_from_iter<I>(iter: I, shape: Vec<usize>) -> BuiltinResult<Value>
where
    I: IntoIterator<Item = bool>,
{
    Ok(Value::LogicalArray(
        LogicalArray::new(iter.into_iter().map(u8::from).collect(), shape)
            .map_err(internal_error)?,
    ))
}

fn any_missing(value: &Value) -> BuiltinResult<bool> {
    match ismissing_value(value)? {
        Value::Bool(flag) => Ok(flag),
        Value::LogicalArray(array) => Ok(array.data.iter().any(|flag| *flag != 0)),
        Value::Struct(st) => {
            for field in st.fields.values() {
                if any_missing(field)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn logical_mask_for_rows(value: &Value, expected_rows: usize) -> BuiltinResult<Vec<u8>> {
    let mask_value = ismissing_value(value)?;
    match mask_value {
        Value::Bool(flag) => Ok(vec![u8::from(flag); expected_rows]),
        Value::LogicalArray(mask) => logical_array_mask_for_rows(&mask, expected_rows),
        _ => Err(unsupported_type("cannot build row missing mask for value")),
    }
}

fn logical_array_mask_for_rows(
    mask: &LogicalArray,
    expected_rows: usize,
) -> BuiltinResult<Vec<u8>> {
    let rows = mask.shape.first().copied().unwrap_or(mask.data.len());
    let cols = mask.shape.get(1).copied().unwrap_or(1);
    if rows == expected_rows {
        let mut out = vec![0u8; expected_rows];
        for col in 0..cols {
            for (row, slot) in out.iter_mut().enumerate().take(expected_rows) {
                let idx = row + col * rows;
                if mask.data.get(idx).copied().unwrap_or(0) != 0 {
                    *slot = 1;
                }
            }
        }
        Ok(out)
    } else if mask.data.len() == expected_rows {
        Ok(mask.data.clone())
    } else {
        Err(invalid_argument(
            "missing mask shape does not match table height",
        ))
    }
}

#[derive(Clone, Copy)]
enum RemoveDim {
    Auto,
    Rows,
    Columns,
}

#[derive(Clone, Copy)]
struct RemoveOptions {
    dim: RemoveDim,
}

impl RemoveOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut dim = RemoveDim::Auto;
        let mut idx = 0;
        while idx < args.len() {
            if let Some(text) = scalar_text(&args[idx]) {
                match text.to_ascii_lowercase().as_str() {
                    "dim" if idx + 1 < args.len() => {
                        dim = dim_from_value(&args[idx + 1])?;
                        idx += 2;
                        continue;
                    }
                    "dim" => return Err(invalid_argument("rmmissing: 'dim' requires a value")),
                    "rows" => {
                        dim = RemoveDim::Rows;
                        idx += 1;
                        continue;
                    }
                    "columns" | "cols" => {
                        dim = RemoveDim::Columns;
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(invalid_argument(format!(
                            "rmmissing: unsupported option '{other}'"
                        )))
                    }
                }
            }
            if matches!(args[idx], Value::Num(_) | Value::Int(_)) {
                dim = dim_from_value(&args[idx])?;
                idx += 1;
                continue;
            }
            return Err(invalid_argument(format!(
                "rmmissing: unsupported option argument {:?}",
                args[idx]
            )));
        }
        Ok(Self { dim })
    }
}

fn dim_from_value(value: &Value) -> BuiltinResult<RemoveDim> {
    match scalar_usize(value, "dimension")? {
        1 => Ok(RemoveDim::Rows),
        2 => Ok(RemoveDim::Columns),
        _ => Err(invalid_argument("dimension must be 1 or 2 for rmmissing")),
    }
}

fn remove_missing_value(
    value: Value,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    match value {
        Value::Object(object) if is_tabular_object(&object) => {
            remove_missing_table(object, options)
        }
        Value::Tensor(tensor) => remove_missing_tensor(tensor, options),
        Value::StringArray(array) => remove_missing_string_array(array, options),
        Value::LogicalArray(array) => remove_missing_logical_array(array, options),
        Value::Cell(cell) => remove_missing_cell(cell, options),
        other => {
            if any_missing(&other)? {
                Ok((
                    empty_like(other)?,
                    LogicalArray::new(vec![1], vec![1, 1]).map_err(internal_error)?,
                ))
            } else {
                Ok((
                    other,
                    LogicalArray::new(vec![0], vec![1, 1]).map_err(internal_error)?,
                ))
            }
        }
    }
}

fn remove_missing_table(
    object: ObjectInstance,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let height = table_height(&object)?;
    let width = table_width(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let remove_columns = matches!(options.dim, RemoveDim::Columns);
    if remove_columns {
        let mut keep_names = Vec::new();
        let mut removed = Vec::with_capacity(width);
        let mut keep_values = Vec::new();
        for name in &names {
            let value = variables
                .fields
                .get(name)
                .ok_or_else(|| internal_error(format!("table missing variable {name}")))?;
            let has_missing = any_missing(value)?;
            removed.push(u8::from(has_missing));
            if !has_missing {
                keep_names.push(name.clone());
                keep_values.push(value.clone());
            }
        }
        let row_names = selected_row_names(&object, &(0..height).collect::<Vec<_>>())?;
        Ok((
            table_from_columns_like(&object, keep_names, keep_values, row_names, None)?,
            LogicalArray::new(removed, vec![1, width]).map_err(internal_error)?,
        ))
    } else {
        let mut row_has_missing = vec![0u8; height];
        for name in &names {
            if let Some(value) = variables.fields.get(name) {
                let mask = logical_mask_for_rows(value, height)?;
                for row in 0..height {
                    if mask[row] != 0 {
                        row_has_missing[row] = 1;
                    }
                }
            }
        }
        let keep_rows: Vec<usize> = row_has_missing
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut values = Vec::with_capacity(names.len());
        for name in &names {
            let value = variables
                .fields
                .get(name)
                .ok_or_else(|| internal_error(format!("table missing variable {name}")))?;
            values.push(select_rows(value, &keep_rows)?);
        }
        let row_names = selected_row_names(&object, &keep_rows)?;
        Ok((
            table_from_columns_like(&object, names, values, row_names, Some(&keep_rows))?,
            LogicalArray::new(row_has_missing, vec![height, 1]).map_err(internal_error)?,
        ))
    }
}

fn remove_missing_tensor(
    tensor: Tensor,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    if tensor.integer_storage().is_some() {
        let rows = tensor.rows();
        let cols = tensor.cols();
        let mask = if rows == 1 || cols == 1 {
            let len = tensor_utils::tensor_element_len(&tensor);
            LogicalArray::new(vec![0; len], vec![1, len])
        } else if matches!(options.dim, RemoveDim::Columns) {
            LogicalArray::new(vec![0; cols], vec![1, cols])
        } else {
            LogicalArray::new(vec![0; rows], vec![rows, 1])
        }
        .map_err(internal_error)?;
        return Ok((Value::Tensor(tensor), mask));
    }
    if tensor.rows() == 1 || tensor.cols() == 1 {
        let dtype = tensor.numeric_dtype();
        let source_rows = tensor.rows();
        let values = tensor_utils::tensor_into_values_f64(tensor);
        let mut data = Vec::new();
        let mut removed = Vec::with_capacity(values.len());
        let source_is_row = source_rows == 1;
        for value in values {
            if value.is_nan() {
                removed.push(1);
            } else {
                removed.push(0);
                data.push(value);
            }
        }
        let shape = if source_is_row {
            vec![1, data.len()]
        } else {
            vec![data.len(), 1]
        };
        let removed_len = removed.len();
        return Ok((
            Value::Tensor(Tensor::new_with_dtype(data, shape, dtype).map_err(internal_error)?),
            LogicalArray::new(removed, vec![1, removed_len]).map_err(internal_error)?,
        ));
    }
    let remove_columns = matches!(options.dim, RemoveDim::Columns);
    if remove_columns {
        remove_missing_columns_tensor(tensor)
    } else {
        remove_missing_rows_tensor(tensor)
    }
}

fn remove_missing_rows_tensor(tensor: Tensor) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut removed = vec![0u8; rows];
    for col in 0..cols {
        for (row, slot) in removed.iter_mut().enumerate().take(rows) {
            if tensor.get2(row, col).map_err(internal_error)?.is_nan() {
                *slot = 1;
            }
        }
    }
    let keep: Vec<usize> = removed
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
        .collect();
    let out = select_rows(&Value::Tensor(tensor), &keep)?;
    Ok((
        out,
        LogicalArray::new(removed, vec![rows, 1]).map_err(internal_error)?,
    ))
}

fn remove_missing_columns_tensor(tensor: Tensor) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut removed = vec![0u8; cols];
    for (col, slot) in removed.iter_mut().enumerate().take(cols) {
        for row in 0..rows {
            if tensor.get2(row, col).map_err(internal_error)?.is_nan() {
                *slot = 1;
            }
        }
    }
    let keep_cols: Vec<usize> = removed
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
        .collect();
    let mut data = Vec::with_capacity(rows * keep_cols.len());
    for col in keep_cols {
        for row in 0..rows {
            data.push(tensor.get2(row, col).map_err(internal_error)?);
        }
    }
    Ok((
        Value::Tensor(
            Tensor::new_with_dtype(
                data,
                vec![rows, removed.iter().filter(|f| **f == 0).count()],
                tensor.numeric_dtype(),
            )
            .map_err(internal_error)?,
        ),
        LogicalArray::new(removed, vec![1, cols]).map_err(internal_error)?,
    ))
}

fn remove_missing_string_array(
    array: StringArray,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = array.rows();
    let cols = array.cols();
    let shape = array.shape.clone();
    remove_missing_column_major(
        array.data,
        rows,
        cols,
        shape,
        options,
        |text| is_missing_text(text),
        |data, shape| {
            StringArray::new(data, shape)
                .map(Value::StringArray)
                .map_err(internal_error)
        },
    )
}

fn remove_missing_logical_array(
    array: LogicalArray,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = array.shape.first().copied().unwrap_or(array.data.len());
    let cols = array.shape.get(1).copied().unwrap_or(1);
    remove_missing_column_major(
        array.data,
        rows,
        cols,
        array.shape.clone(),
        options,
        |_| false,
        |data, shape| {
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(internal_error)
        },
    )
}

fn remove_missing_cell(
    cell: CellArray,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = cell.rows;
    let cols = cell.cols;
    let is_missing = |row: usize, col: usize| -> bool {
        cell.get(row, col)
            .ok()
            .and_then(|value| any_missing(&value).ok())
            .unwrap_or(false)
    };

    if rows == 1 || cols == 1 {
        let mut out = Vec::new();
        let mut removed = Vec::with_capacity(cell.data.len());
        for value in cell.data {
            if any_missing(&value).unwrap_or(false) {
                removed.push(1);
            } else {
                removed.push(0);
                out.push(value);
            }
        }
        let out_shape = if rows == 1 {
            vec![1, out.len()]
        } else {
            vec![out.len(), 1]
        };
        let removed_len = removed.len();
        let out_rows = out_shape.first().copied().unwrap_or(0);
        let out_cols = out_shape.get(1).copied().unwrap_or(0);
        return Ok((
            CellArray::new(out, out_rows, out_cols)
                .map(Value::Cell)
                .map_err(internal_error)?,
            LogicalArray::new(removed, vec![1, removed_len]).map_err(internal_error)?,
        ));
    }

    if matches!(options.dim, RemoveDim::Columns) {
        let mut removed = vec![0u8; cols];
        for col in 0..cols {
            for row in 0..rows {
                if is_missing(row, col) {
                    removed[col] = 1;
                }
            }
        }
        let kept_cols: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(rows * kept_cols.len());
        for row in 0..rows {
            for col in &kept_cols {
                out.push(cell.get(row, *col).map_err(internal_error)?);
            }
        }
        let kept = kept_cols.len();
        Ok((
            CellArray::new(out, rows, kept)
                .map(Value::Cell)
                .map_err(internal_error)?,
            LogicalArray::new(removed, vec![1, cols]).map_err(internal_error)?,
        ))
    } else {
        let mut removed = vec![0u8; rows];
        for row in 0..rows {
            for col in 0..cols {
                if is_missing(row, col) {
                    removed[row] = 1;
                }
            }
        }
        let kept_rows: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(kept_rows.len() * cols);
        for row in &kept_rows {
            for col in 0..cols {
                out.push(cell.get(*row, col).map_err(internal_error)?);
            }
        }
        Ok((
            CellArray::new(out, kept_rows.len(), cols)
                .map(Value::Cell)
                .map_err(internal_error)?,
            LogicalArray::new(removed, vec![rows, 1]).map_err(internal_error)?,
        ))
    }
}

fn remove_missing_column_major<T: Clone>(
    data: Vec<T>,
    rows: usize,
    cols: usize,
    _shape: Vec<usize>,
    options: RemoveOptions,
    is_missing: impl Fn(&T) -> bool,
    build: impl Fn(Vec<T>, Vec<usize>) -> BuiltinResult<Value>,
) -> BuiltinResult<(Value, LogicalArray)> {
    if rows == 1 || cols == 1 {
        let mut out = Vec::new();
        let mut removed = Vec::with_capacity(data.len());
        for value in data {
            if is_missing(&value) {
                removed.push(1);
            } else {
                removed.push(0);
                out.push(value);
            }
        }
        let out_shape = if rows == 1 {
            vec![1, out.len()]
        } else {
            vec![out.len(), 1]
        };
        let removed_len = removed.len();
        return Ok((
            build(out, out_shape)?,
            LogicalArray::new(removed, vec![1, removed_len]).map_err(internal_error)?,
        ));
    }
    if matches!(options.dim, RemoveDim::Columns) {
        let mut removed = vec![0u8; cols];
        for col in 0..cols {
            for row in 0..rows {
                if is_missing(&data[row + col * rows]) {
                    removed[col] = 1;
                }
            }
        }
        let kept_cols: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(rows * kept_cols.len());
        for col in kept_cols {
            for row in 0..rows {
                out.push(data[row + col * rows].clone());
            }
        }
        let kept = removed.iter().filter(|flag| **flag == 0).count();
        Ok((
            build(out, vec![rows, kept])?,
            LogicalArray::new(removed, vec![1, cols]).map_err(internal_error)?,
        ))
    } else {
        let mut removed = vec![0u8; rows];
        for col in 0..cols {
            for row in 0..rows {
                if is_missing(&data[row + col * rows]) {
                    removed[row] = 1;
                }
            }
        }
        let kept_rows: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(kept_rows.len() * cols);
        for col in 0..cols {
            for row in &kept_rows {
                out.push(data[*row + col * rows].clone());
            }
        }
        Ok((
            build(out, vec![kept_rows.len(), cols])?,
            LogicalArray::new(removed, vec![rows, 1]).map_err(internal_error)?,
        ))
    }
}

fn empty_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            Tensor::new_with_dtype(Vec::new(), vec![0, 0], tensor.numeric_dtype())
                .map(Value::Tensor)
                .map_err(internal_error)
        }
        Value::StringArray(_) | Value::String(_) => StringArray::new(Vec::new(), vec![0, 0])
            .map(Value::StringArray)
            .map_err(internal_error),
        Value::Cell(_) => CellArray::new(Vec::new(), 0, 0)
            .map(Value::Cell)
            .map_err(internal_error),
        _ => Ok(Value::OutputList(Vec::new())),
    }
}

#[derive(Clone)]
enum FillMethod {
    Constant(Value),
    Previous,
    Next,
    Nearest,
    Linear,
    Mean,
    Median,
}

#[derive(Clone)]
struct FillOptions {
    method: FillMethod,
    dim: Option<usize>,
}

impl FillOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        if args.is_empty() {
            return Err(invalid_argument("fillmissing: method is required"));
        }
        let mut idx = 0;
        let method_text = scalar_text(&args[idx])
            .ok_or_else(|| invalid_argument("fillmissing: method must be a string"))?
            .to_ascii_lowercase();
        idx += 1;
        let method = match method_text.as_str() {
            "constant" => {
                let fill = args
                    .get(idx)
                    .ok_or_else(|| {
                        invalid_argument("fillmissing: constant method needs a fill value")
                    })?
                    .clone();
                idx += 1;
                FillMethod::Constant(fill)
            }
            "previous" => FillMethod::Previous,
            "next" => FillMethod::Next,
            "nearest" => FillMethod::Nearest,
            "linear" => FillMethod::Linear,
            "mean" => FillMethod::Mean,
            "median" => FillMethod::Median,
            other => {
                return Err(invalid_argument(format!(
                    "fillmissing: unsupported method '{other}'"
                )))
            }
        };
        let mut dim = None;
        while idx < args.len() {
            if let Some(text) = scalar_text(&args[idx]) {
                if text.eq_ignore_ascii_case("dim") && idx + 1 < args.len() {
                    dim = Some(scalar_usize(&args[idx + 1], "fillmissing dimension")?);
                    idx += 2;
                    continue;
                }
                if text.eq_ignore_ascii_case("dim") {
                    return Err(invalid_argument("fillmissing: 'dim' requires a value"));
                }
                return Err(invalid_argument(format!(
                    "fillmissing: unsupported option '{text}'"
                )));
            }
            if matches!(args[idx], Value::Num(_) | Value::Int(_)) {
                dim = Some(scalar_usize(&args[idx], "fillmissing dimension")?);
                idx += 1;
                continue;
            }
            return Err(invalid_argument(format!(
                "fillmissing: unsupported option argument {:?}",
                args[idx]
            )));
        }
        if dim.is_some_and(|dim| dim != 1 && dim != 2) {
            return Err(invalid_argument("fillmissing: dimension must be 1 or 2"));
        }
        Ok(Self { method, dim })
    }
}

fn fill_missing_value(value: Value, options: &FillOptions) -> BuiltinResult<(Value, LogicalArray)> {
    match value {
        Value::Tensor(tensor) => fill_missing_tensor(tensor, options),
        Value::StringArray(array) => fill_missing_string_array(array, options),
        Value::Object(object) if is_tabular_object(&object) => fill_missing_table(object, options),
        Value::Cell(cell) => fill_missing_cell(cell, options),
        Value::ComplexTensor(_) => Err(unsupported_type(
            "fillmissing: complex arrays are not supported yet",
        )),
        Value::SparseTensor(_) => Err(unsupported_type(
            "fillmissing: sparse arrays are not supported yet",
        )),
        other => {
            let missing = any_missing(&other)?;
            let mask =
                LogicalArray::new(vec![u8::from(missing)], vec![1, 1]).map_err(internal_error)?;
            if missing {
                match &options.method {
                    FillMethod::Constant(fill) => Ok((fill.clone(), mask)),
                    _ => Err(unsupported_type(
                        "fillmissing: scalar fill requires the constant method",
                    )),
                }
            } else {
                Ok((other, mask))
            }
        }
    }
}

fn fill_missing_table(
    mut object: ObjectInstance,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let height = table_height(&object)?;
    let width = table_width(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut output_vars = StructValue::new();
    let mut mask_data = vec![0u8; height * width];
    for (col, name) in names.iter().enumerate() {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| internal_error(format!("table missing variable {name}")))?
            .clone();
        let (filled, mask) = fill_missing_value(value, options)?;
        let row_mask = logical_array_mask_for_rows(&mask, height)?;
        for row in 0..height {
            if row_mask.get(row).copied().unwrap_or(0) != 0 {
                mask_data[row + col * height] = 1;
            }
        }
        output_vars.insert(name.clone(), filled);
    }
    object
        .properties
        .insert("__table_variables".to_string(), Value::Struct(output_vars));
    Ok((
        Value::Object(object),
        LogicalArray::new(mask_data, vec![height, width]).map_err(internal_error)?,
    ))
}

fn fill_missing_tensor(
    tensor: Tensor,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    if tensor.integer_storage().is_some() {
        return Ok((
            Value::Tensor(tensor),
            LogicalArray::new(vec![0; rows * cols], vec![rows, cols]).map_err(internal_error)?,
        ));
    }
    let dim = options
        .dim
        .unwrap_or_else(|| first_nonsingleton_dim(rows, cols));
    validate_matrix_dim(dim, "fillmissing")?;
    let dtype = tensor.numeric_dtype();
    let shape = tensor.shape.clone();
    let mut data = tensor_utils::tensor_into_values_f64(tensor);
    let mut mask: Vec<u8> = data.iter().map(|value| u8::from(value.is_nan())).collect();
    match &options.method {
        FillMethod::Constant(fill) => {
            let fill = numeric_scalar(fill, "fillmissing constant")?;
            for (idx, value) in data.iter_mut().enumerate() {
                if mask[idx] != 0 {
                    *value = fill;
                }
            }
        }
        FillMethod::Mean => fill_summary_numeric(&mut data, rows, cols, dim, Summary::Mean),
        FillMethod::Median => fill_summary_numeric(&mut data, rows, cols, dim, Summary::Median),
        FillMethod::Previous => {
            fill_neighbor_numeric(&mut data, rows, cols, dim, Neighbor::Previous)
        }
        FillMethod::Next => fill_neighbor_numeric(&mut data, rows, cols, dim, Neighbor::Next),
        FillMethod::Nearest => fill_nearest_numeric(&mut data, rows, cols, dim),
        FillMethod::Linear => fill_linear_numeric(&mut data, rows, cols, dim),
    }
    for (flag, value) in mask.iter_mut().zip(&data) {
        if value.is_nan() {
            *flag = 0;
        }
    }
    Ok((
        Value::Tensor(Tensor::new_with_dtype(data, shape, dtype).map_err(internal_error)?),
        LogicalArray::new(mask, vec![rows, cols]).map_err(internal_error)?,
    ))
}

fn fill_missing_string_array(
    array: StringArray,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = array.rows();
    let cols = array.cols();
    let dim = options
        .dim
        .unwrap_or_else(|| first_nonsingleton_dim(rows, cols));
    validate_matrix_dim(dim, "fillmissing")?;
    let mut data = array.data.clone();
    let mut mask: Vec<u8> = data
        .iter()
        .map(|text| u8::from(is_missing_text(text)))
        .collect();
    match &options.method {
        FillMethod::Constant(fill) => {
            let fill = scalar_text(fill)
                .ok_or_else(|| invalid_argument("fillmissing: string constant must be text"))?;
            for (idx, text) in data.iter_mut().enumerate() {
                if mask[idx] != 0 {
                    *text = fill.clone();
                }
            }
        }
        FillMethod::Previous => fill_neighbor_text(&mut data, rows, cols, dim, Neighbor::Previous),
        FillMethod::Next => fill_neighbor_text(&mut data, rows, cols, dim, Neighbor::Next),
        FillMethod::Nearest => fill_nearest_text(&mut data, rows, cols, dim),
        _ => {
            return Err(unsupported_type(
                "fillmissing: string arrays support constant, previous, next, and nearest",
            ))
        }
    }
    for (flag, text) in mask.iter_mut().zip(&data) {
        if is_missing_text(text) {
            *flag = 0;
        }
    }
    Ok((
        Value::StringArray(StringArray::new(data, array.shape).map_err(internal_error)?),
        LogicalArray::new(mask, vec![rows, cols]).map_err(internal_error)?,
    ))
}

fn fill_missing_cell(
    cell: CellArray,
    options: &FillOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let mut data = cell.data.clone();
    let mut mask = Vec::with_capacity(data.len());
    for item in &data {
        mask.push(u8::from(any_missing(item)?));
    }
    match &options.method {
        FillMethod::Constant(fill) => {
            for (idx, item) in data.iter_mut().enumerate() {
                if mask[idx] != 0 {
                    *item = fill.clone();
                }
            }
        }
        _ => {
            return Err(unsupported_type(
                "fillmissing: cell arrays currently support the constant method",
            ))
        }
    }
    for (flag, item) in mask.iter_mut().zip(&data) {
        if any_missing(item)? {
            *flag = 0;
        }
    }
    Ok((
        Value::Cell(CellArray::new(data, cell.rows, cell.cols).map_err(internal_error)?),
        LogicalArray::new(mask, vec![cell.rows, cell.cols]).map_err(internal_error)?,
    ))
}

enum Summary {
    Mean,
    Median,
}

fn fill_summary_numeric(data: &mut [f64], rows: usize, cols: usize, dim: usize, summary: Summary) {
    if dim == 1 {
        for col in 0..cols {
            let vals = finite_slice(data, rows, col, true);
            let replacement = summary_value(vals, &summary);
            for row in 0..rows {
                let idx = row + col * rows;
                if data[idx].is_nan() {
                    data[idx] = replacement;
                }
            }
        }
    } else {
        for row in 0..rows {
            let vals = finite_slice(data, rows, row, false);
            let replacement = summary_value(vals, &summary);
            for col in 0..cols {
                let idx = row + col * rows;
                if data[idx].is_nan() {
                    data[idx] = replacement;
                }
            }
        }
    }
}

fn finite_slice(data: &[f64], rows: usize, fixed: usize, along_rows: bool) -> Vec<f64> {
    let mut out = Vec::new();
    if along_rows {
        let cols = data.len() / rows;
        for row in 0..rows {
            let value = data[row + fixed * rows];
            if !value.is_nan() {
                out.push(value);
            }
        }
        debug_assert!(fixed < cols);
    } else {
        let cols = data.len() / rows;
        for col in 0..cols {
            let value = data[fixed + col * rows];
            if !value.is_nan() {
                out.push(value);
            }
        }
    }
    out
}

fn summary_value(mut vals: Vec<f64>, summary: &Summary) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    match summary {
        Summary::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
        Summary::Median => {
            vals.sort_by(|a, b| a.total_cmp(b));
            let mid = vals.len() / 2;
            if vals.len().is_multiple_of(2) {
                (vals[mid - 1] + vals[mid]) / 2.0
            } else {
                vals[mid]
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Neighbor {
    Previous,
    Next,
}

fn fill_neighbor_numeric(data: &mut [f64], rows: usize, cols: usize, dim: usize, dir: Neighbor) {
    if dim == 1 {
        for col in 0..cols {
            fill_line_numeric(data, rows, col, rows, 1, dir);
        }
    } else {
        for row in 0..rows {
            fill_line_numeric(data, rows, row, cols, rows, dir);
        }
    }
}

fn fill_line_numeric(
    data: &mut [f64],
    _rows: usize,
    start: usize,
    len: usize,
    step: usize,
    dir: Neighbor,
) {
    match dir {
        Neighbor::Previous => {
            let mut last = None;
            for i in 0..len {
                let idx = start + i * step;
                if data[idx].is_nan() {
                    if let Some(value) = last {
                        data[idx] = value;
                    }
                } else {
                    last = Some(data[idx]);
                }
            }
        }
        Neighbor::Next => {
            let mut next = None;
            for i in (0..len).rev() {
                let idx = start + i * step;
                if data[idx].is_nan() {
                    if let Some(value) = next {
                        data[idx] = value;
                    }
                } else {
                    next = Some(data[idx]);
                }
            }
        }
    }
}

fn fill_nearest_numeric(data: &mut [f64], rows: usize, cols: usize, dim: usize) {
    let original = data.to_vec();
    if dim == 1 {
        for col in 0..cols {
            fill_nearest_line_numeric(&original, data, col * rows, rows, 1);
        }
    } else {
        for row in 0..rows {
            fill_nearest_line_numeric(&original, data, row, cols, rows);
        }
    }
}

fn fill_nearest_line_numeric(
    original: &[f64],
    data: &mut [f64],
    start: usize,
    len: usize,
    step: usize,
) {
    for i in 0..len {
        let idx = start + i * step;
        if !original[idx].is_nan() {
            continue;
        }
        let prev = (0..i).rev().find_map(|j| {
            let value = original[start + j * step];
            (!value.is_nan()).then_some((i - j, value))
        });
        let next = ((i + 1)..len).find_map(|j| {
            let value = original[start + j * step];
            (!value.is_nan()).then_some((j - i, value))
        });
        data[idx] = match (prev, next) {
            (Some((pd, _)), Some((nd, nv))) if nd < pd => nv,
            (Some((_, pv)), _) => pv,
            (_, Some((_, nv))) => nv,
            _ => f64::NAN,
        };
    }
}

fn fill_linear_numeric(data: &mut [f64], rows: usize, cols: usize, dim: usize) {
    if dim == 1 {
        for col in 0..cols {
            fill_linear_line(data, col * rows, rows, 1);
        }
    } else {
        for row in 0..rows {
            fill_linear_line(data, row, cols, rows);
        }
    }
}

fn fill_linear_line(data: &mut [f64], start: usize, len: usize, step: usize) {
    let mut i = 0;
    while i < len {
        let idx = start + i * step;
        if !data[idx].is_nan() {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < len && data[start + i * step].is_nan() {
            i += 1;
        }
        let run_end = i;
        let prev = (run_start > 0).then(|| data[start + (run_start - 1) * step]);
        let next = (run_end < len).then(|| data[start + run_end * step]);
        match (prev, next) {
            (Some(a), Some(b)) if !a.is_nan() && !b.is_nan() => {
                let span = (run_end - run_start + 1) as f64;
                for (offset, pos) in (run_start..run_end).enumerate() {
                    data[start + pos * step] = a + (b - a) * ((offset + 1) as f64 / span);
                }
            }
            (Some(a), _) if !a.is_nan() => {
                for pos in run_start..run_end {
                    data[start + pos * step] = a;
                }
            }
            (_, Some(b)) if !b.is_nan() => {
                for pos in run_start..run_end {
                    data[start + pos * step] = b;
                }
            }
            _ => {}
        }
    }
}

fn fill_neighbor_text(data: &mut [String], rows: usize, cols: usize, dim: usize, dir: Neighbor) {
    if dim == 1 {
        for col in 0..cols {
            fill_line_text(data, col * rows, rows, 1, dir);
        }
    } else {
        for row in 0..rows {
            fill_line_text(data, row, cols, rows, dir);
        }
    }
}

fn fill_line_text(data: &mut [String], start: usize, len: usize, step: usize, dir: Neighbor) {
    match dir {
        Neighbor::Previous => {
            let mut last: Option<String> = None;
            for i in 0..len {
                let idx = start + i * step;
                if is_missing_text(&data[idx]) {
                    if let Some(value) = &last {
                        data[idx] = value.clone();
                    }
                } else {
                    last = Some(data[idx].clone());
                }
            }
        }
        Neighbor::Next => {
            let mut next: Option<String> = None;
            for i in (0..len).rev() {
                let idx = start + i * step;
                if is_missing_text(&data[idx]) {
                    if let Some(value) = &next {
                        data[idx] = value.clone();
                    }
                } else {
                    next = Some(data[idx].clone());
                }
            }
        }
    }
}

fn fill_nearest_text(data: &mut [String], rows: usize, cols: usize, dim: usize) {
    let original = data.to_vec();
    if dim == 1 {
        for col in 0..cols {
            fill_nearest_line_text(&original, data, col * rows, rows, 1);
        }
    } else {
        for row in 0..rows {
            fill_nearest_line_text(&original, data, row, cols, rows);
        }
    }
}

fn fill_nearest_line_text(
    original: &[String],
    data: &mut [String],
    start: usize,
    len: usize,
    step: usize,
) {
    for i in 0..len {
        let idx = start + i * step;
        if !is_missing_text(&original[idx]) {
            continue;
        }
        let prev = (0..i).rev().find_map(|j| {
            let value = &original[start + j * step];
            (!is_missing_text(value)).then_some((i - j, value.clone()))
        });
        let next = ((i + 1)..len).find_map(|j| {
            let value = &original[start + j * step];
            (!is_missing_text(value)).then_some((j - i, value.clone()))
        });
        data[idx] = match (prev, next) {
            (Some((pd, _)), Some((nd, nv))) if nd < pd => nv,
            (Some((_, pv)), _) => pv,
            (_, Some((_, nv))) => nv,
            _ => MISSING_TEXT.to_string(),
        };
    }
}

#[derive(Clone, Copy)]
struct MovingOptions {
    dim: Option<usize>,
    omit_nan: bool,
}

impl MovingOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut dim = None;
        let mut omit_nan = false;
        let mut idx = 0;
        while idx < args.len() {
            if let Some(text) = scalar_text(&args[idx]) {
                match text.to_ascii_lowercase().as_str() {
                    "omitnan" | "omitmissing" => omit_nan = true,
                    "includenan" | "includemissing" => omit_nan = false,
                    "dim" if idx + 1 < args.len() => {
                        dim = Some(scalar_usize(&args[idx + 1], "movmad dimension")?);
                        idx += 1;
                    }
                    "dim" => return Err(invalid_argument("movmad: 'dim' requires a value")),
                    other => {
                        return Err(invalid_argument(format!(
                            "movmad: unsupported option '{other}'"
                        )))
                    }
                }
            } else if matches!(args[idx], Value::Num(_) | Value::Int(_)) {
                dim = Some(scalar_usize(&args[idx], "movmad dimension")?);
            } else {
                return Err(invalid_argument(format!(
                    "movmad: unsupported option argument {:?}",
                    args[idx]
                )));
            }
            idx += 1;
        }
        if dim.is_some_and(|dim| dim != 1 && dim != 2) {
            return Err(invalid_argument("movmad: dimension must be 1 or 2"));
        }
        Ok(Self { dim, omit_nan })
    }
}

fn moving_mad(tensor: Tensor, window: usize, options: MovingOptions) -> BuiltinResult<Value> {
    if window == 0 {
        return Err(invalid_argument("movmad: window length must be positive"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let dim = options
        .dim
        .unwrap_or_else(|| first_nonsingleton_dim(rows, cols));
    validate_matrix_dim(dim, "movmad")?;
    let shape = tensor.shape.clone();
    let output_dtype = if tensor.integer_storage().is_some() {
        NumericDType::F64
    } else {
        tensor.numeric_dtype()
    };
    let values = tensor_utils::tensor_values_f64_cow(&tensor);
    let mut out = vec![f64::NAN; values.len()];
    if dim == 1 {
        for col in 0..cols {
            for row in 0..rows {
                out[row + col * rows] = moving_mad_at(
                    &values,
                    rows,
                    col * rows,
                    row,
                    rows,
                    1,
                    window,
                    options.omit_nan,
                );
            }
        }
    } else {
        for row in 0..rows {
            for col in 0..cols {
                out[row + col * rows] = moving_mad_at(
                    &values,
                    rows,
                    row,
                    col,
                    cols,
                    rows,
                    window,
                    options.omit_nan,
                );
            }
        }
    }
    Tensor::new_with_dtype(out, shape, output_dtype)
        .map(Value::Tensor)
        .map_err(internal_error)
}

fn moving_mad_at(
    data: &[f64],
    _rows: usize,
    start: usize,
    pos: usize,
    len: usize,
    step: usize,
    window: usize,
    omit_nan: bool,
) -> f64 {
    let before = (window - 1) / 2;
    let after = window / 2;
    let lo = pos.saturating_sub(before);
    let hi = pos.saturating_add(after).saturating_add(1).min(len);
    let mut vals = Vec::new();
    for idx in lo..hi {
        let value = data[start + idx * step];
        if value.is_nan() && omit_nan {
            continue;
        }
        vals.push(value);
    }
    if vals.is_empty() || vals.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let med = summary_value(vals.clone(), &Summary::Median);
    let mut devs: Vec<f64> = vals.into_iter().map(|value| (value - med).abs()).collect();
    summary_value(::std::mem::take(&mut devs), &Summary::Median)
}

struct IndicatorSet {
    numeric: Vec<f64>,
    text: Vec<String>,
}

fn indicator_set(value: &Value) -> BuiltinResult<IndicatorSet> {
    let mut set = IndicatorSet {
        numeric: Vec::new(),
        text: Vec::new(),
    };
    collect_indicators(value, &mut set)?;
    Ok(set)
}

fn collect_indicators(value: &Value, set: &mut IndicatorSet) -> BuiltinResult<()> {
    match value {
        Value::Num(n) => set.numeric.push(*n),
        Value::Int(i) => set.numeric.push(i.to_f64()),
        Value::String(s) => set.text.push(s.clone()),
        Value::StringArray(array) => set.text.extend(array.data.iter().cloned()),
        Value::CharArray(array) => set.text.extend(char_rows(array)),
        Value::Tensor(tensor) => set.numeric.extend(tensor_utils::tensor_values_f64(tensor)),
        Value::Cell(cell) => {
            for item in &cell.data {
                collect_indicators(item, set)?;
            }
        }
        other => {
            return Err(unsupported_type(format!(
                "unsupported missing indicator {other:?}"
            )))
        }
    }
    Ok(())
}

fn standardize_missing_value(value: Value, indicators: &IndicatorSet) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => Ok(Value::Tensor(tensor)),
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            let shape = tensor.shape.clone();
            let mut values = tensor_utils::tensor_into_values_f64(tensor);
            for value in &mut values {
                if indicators
                    .numeric
                    .iter()
                    .any(|marker| numeric_indicator_matches(*value, *marker))
                {
                    *value = f64::NAN;
                }
            }
            Tensor::new_with_dtype(values, shape, dtype)
                .map(Value::Tensor)
                .map_err(internal_error)
        }
        Value::String(mut s) => {
            if indicators.text.iter().any(|marker| marker == &s) {
                s = MISSING_TEXT.to_string();
            }
            Ok(Value::String(s))
        }
        Value::StringArray(mut array) => {
            for text in &mut array.data {
                if indicators.text.iter().any(|marker| marker == text) {
                    *text = MISSING_TEXT.to_string();
                }
            }
            Ok(Value::StringArray(array))
        }
        Value::CharArray(array) => {
            let rows = char_rows(&array);
            let data: Vec<String> = rows
                .into_iter()
                .map(|text| {
                    if indicators.text.iter().any(|marker| marker == &text) {
                        MISSING_TEXT.to_string()
                    } else {
                        text
                    }
                })
                .collect();
            Ok(Value::StringArray(
                StringArray::new(data, vec![array.rows, 1]).map_err(internal_error)?,
            ))
        }
        Value::Object(mut object) if is_tabular_object(&object) => {
            let variables = table_variables(&object)?;
            let mut out = StructValue::new();
            for (name, field) in variables.fields {
                out.insert(name, standardize_missing_value(field, indicators)?);
            }
            object
                .properties
                .insert("__table_variables".to_string(), Value::Struct(out));
            Ok(Value::Object(object))
        }
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                out.push(standardize_missing_value(item, indicators)?);
            }
            Ok(Value::Cell(
                CellArray::new(out, cell.rows, cell.cols).map_err(internal_error)?,
            ))
        }
        other => Ok(other),
    }
}

fn numeric_indicator_matches(value: f64, marker: f64) -> bool {
    if marker.is_nan() {
        value.is_nan()
    } else {
        value == marker
    }
}

fn numeric_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(internal_error),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(internal_error),
        Value::LogicalArray(array) => Tensor::new(
            array
                .data
                .iter()
                .map(|flag| f64::from(*flag != 0))
                .collect(),
            array.shape,
        )
        .map_err(internal_error),
        other => Err(unsupported_type(format!(
            "{context}: expected numeric input, got {other:?}"
        ))),
    }
}

fn is_numeric_data_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    )
}

fn pairwise_nan_min(left: Value, right: Value) -> BuiltinResult<Value> {
    if crate::builtins::math::reduction::integer_native::value_has_integer_storage(&left)
        || crate::builtins::math::reduction::integer_native::value_has_integer_storage(&right)
    {
        if let Ok(Some(evaluation)) =
            crate::builtins::math::reduction::integer_native::elementwise_value_extrema(
                &left,
                &right,
                crate::builtins::math::reduction::integer_native::ExtremaDirection::Min,
                crate::builtins::math::reduction::integer_native::ExtremaComparison::Natural,
                false,
            )
        {
            return Ok(evaluation.values);
        }
    }
    let left_scalar = numeric_scalar(&left, "nanmin left").ok();
    let right_scalar = numeric_scalar(&right, "nanmin right").ok();
    if let (Some(a), Some(b)) = (left_scalar, right_scalar) {
        return Ok(Value::Num(nan_min_pair(a, b)));
    }
    let left = numeric_tensor(left, "nanmin left")?;
    let right = numeric_tensor(right, "nanmin right")?;
    let (data, shape, dtype) = broadcast_pairwise_numeric(&left, &right, nan_min_pair)?;
    Tensor::new_with_dtype(data, shape, dtype)
        .map(Value::Tensor)
        .map_err(internal_error)
}

fn broadcast_pairwise_numeric(
    left: &Tensor,
    right: &Tensor,
    op: impl Fn(f64, f64) -> f64,
) -> BuiltinResult<(Vec<f64>, Vec<usize>, runmat_builtins::NumericDType)> {
    let left_len = tensor_utils::tensor_element_len(left);
    let right_len = tensor_utils::tensor_element_len(right);
    if left_len == right_len && left.shape == right.shape {
        let left_values = tensor_utils::tensor_values_f64_cow(left);
        let right_values = tensor_utils::tensor_values_f64_cow(right);
        let data = left_values
            .iter()
            .zip(right_values.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();
        return Ok((data, left.shape.clone(), left.numeric_dtype()));
    }
    if left_len == 1 {
        let left_values = tensor_utils::tensor_values_f64_cow(left);
        let right_values = tensor_utils::tensor_values_f64_cow(right);
        let data = right_values
            .iter()
            .map(|b| op(left_values[0], *b))
            .collect();
        return Ok((data, right.shape.clone(), right.numeric_dtype()));
    }
    if right_len == 1 {
        let left_values = tensor_utils::tensor_values_f64_cow(left);
        let right_values = tensor_utils::tensor_values_f64_cow(right);
        let data = left_values
            .iter()
            .map(|a| op(*a, right_values[0]))
            .collect();
        return Ok((data, left.shape.clone(), left.numeric_dtype()));
    }
    Err(invalid_argument(
        "nanmin: pairwise inputs must have the same shape or one scalar input",
    ))
}

fn nan_min_pair(a: f64, b: f64) -> f64 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => f64::NAN,
        (true, false) => b,
        (false, true) => a,
        (false, false) => a.min(b),
    }
}

fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(f64::from(*b)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        other => Err(invalid_argument(format!(
            "{context}: expected numeric scalar, got {other:?}"
        ))),
    }
}

fn first_nonsingleton_dim(rows: usize, cols: usize) -> usize {
    if rows > 1 {
        1
    } else if cols > 1 {
        2
    } else {
        1
    }
}

fn validate_matrix_dim(dim: usize, context: &str) -> BuiltinResult<()> {
    if dim == 1 || dim == 2 {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "{context}: dimension must be 1 or 2"
        )))
    }
}

fn scalar_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(integer) => return integer_size_to_usize(integer, context),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let integer = storage.value_at(0).ok_or_else(|| {
                    internal_error(format!("{context}: integer scalar storage length mismatch"))
                })?;
                return integer_size_to_usize(&integer, context);
            }
        }
        _ => {}
    }
    let n = numeric_scalar(value, context)?;
    numeric_size_to_usize(n, context)
}

fn integer_size_to_usize(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        invalid_argument(format!("{context}: expected nonnegative platform integer"))
    })
}

fn numeric_size_to_usize(n: f64, context: &str) -> BuiltinResult<usize> {
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return Err(invalid_argument(format!(
            "{context}: expected nonnegative integer"
        )));
    }
    if n > usize::MAX as f64 {
        return Err(invalid_argument(format!("{context}: integer too large")));
    }
    if usize::BITS == 64 && n == usize::MAX as f64 {
        return Err(invalid_argument(format!("{context}: integer too large")));
    }
    Ok(n as usize)
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
}

fn char_rows(array: &CharArray) -> Vec<String> {
    let mut out = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        out.push(array.data[start..start + array.cols].iter().collect());
    }
    out
}

fn is_missing_text(text: &str) -> bool {
    text.eq_ignore_ascii_case(MISSING_TEXT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn first_unrepresentable_usize_double() -> f64 {
        if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        }
    }

    #[test]
    fn missing_constructs_scalar_and_arrays() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let scalar = block_on(missing_builtin(Vec::new())).unwrap();
        assert!(matches!(scalar, Value::StringArray(sa) if sa.data == vec![MISSING_TEXT]));
        let shaped = block_on(missing_builtin(vec![Value::Num(2.0), Value::Num(3.0)])).unwrap();
        assert!(
            matches!(shaped, Value::StringArray(sa) if sa.shape == vec![2, 3] && sa.data.len() == 6)
        );
    }

    #[test]
    fn missing_runmat_shape_extension_reads_every_integer_class_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = [
            IntValue::I8(2),
            IntValue::I16(2),
            IntValue::I32(2),
            IntValue::I64(2),
            IntValue::U8(2),
            IntValue::U16(2),
            IntValue::U32(2),
            IntValue::U64(2),
        ];
        for size in cases {
            let result =
                block_on(missing_builtin(vec![Value::Int(size)])).expect("RunMat shaped missing");
            assert!(
                matches!(result, Value::StringArray(array) if array.shape == vec![2, 2] && array.data.len() == 4)
            );
        }
    }

    #[test]
    fn missing_shaped_extension_gates_before_provider_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let value = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_419_005,
        });
        let error = block_on(missing_builtin(vec![value]))
            .expect_err("MATLAB-compatible mode must reject shaped missing");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MissingShapedArrayExtension")
        );
    }

    #[test]
    fn missing_preserves_typed_integer_size_vectors_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let large = 9_007_199_254_740_993_u64;
        let dims = Tensor::new_integer(IntegerStorage::U64(vec![large, 0]), vec![1, 2]).unwrap();

        let result = block_on(missing_builtin(vec![Value::Tensor(dims)])).unwrap();

        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![large as usize, 0]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn missing_parses_typed_integer_scalar_tensors_exactly() {
        let scalar = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );

        assert_eq!(
            scalar_usize(&scalar, "missing size").unwrap(),
            9_007_199_254_740_993
        );
    }

    #[test]
    fn missing_numeric_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
            .expect("typed numeric scalar");

        assert_eq!(
            numeric_scalar(&Value::Tensor(tensor), "fillmissing constant").unwrap(),
            2026.0
        );
    }

    #[test]
    fn missing_rejects_negative_typed_integer_sizes() {
        let scalar = Tensor::new_integer(IntegerStorage::I8(vec![-1]), vec![1, 1]).unwrap();
        assert!(scalar_usize(&Value::Tensor(scalar), "missing size").is_err());

        let dims = Tensor::new_integer(IntegerStorage::I64(vec![2, -1]), vec![1, 2]).unwrap();
        assert!(tensor_shape_as_size(&dims).is_err());
    }

    #[test]
    fn missing_rejects_unrepresentable_double_sizes_before_casting() {
        let err = scalar_usize(
            &Value::Num(first_unrepresentable_usize_double()),
            "missing size",
        )
        .unwrap_err();
        assert!(err.message.contains("integer too large"), "{}", err.message);

        let dims =
            Tensor::new(vec![first_unrepresentable_usize_double(), 0.0], vec![1, 2]).unwrap();
        let err = tensor_shape_as_size(&dims).unwrap_err();
        assert!(err.message.contains("platform limits"), "{}", err.message);
    }

    #[test]
    fn ismissing_detects_numeric_and_string_values() {
        let result = block_on(ismissing_builtin(tensor(
            vec![1.0, f64::NAN, 3.0],
            vec![1, 3],
        )))
        .unwrap();
        assert!(matches!(result, Value::LogicalArray(mask) if mask.data == vec![0, 1, 0]));

        let strings = StringArray::new(vec!["a".into(), MISSING_TEXT.into()], vec![1, 2]).unwrap();
        let result = block_on(ismissing_builtin(Value::StringArray(strings))).unwrap();
        assert!(matches!(result, Value::LogicalArray(mask) if mask.data == vec![0, 1]));
    }

    #[test]
    fn ismissing_typed_integer_tensor_ignores_f64_mirror() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3])
            .expect("integer tensor");

        let result = block_on(ismissing_builtin(Value::Tensor(input))).unwrap();

        assert!(matches!(
            result,
            Value::LogicalArray(mask) if mask.data == vec![0, 0, 0] && mask.shape == vec![1, 3]
        ));
    }

    #[test]
    fn ismissing_returns_same_shaped_false_for_all_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
            IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
            IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
            IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
        ];
        for storage in storages {
            let input = Tensor::new_integer(storage, vec![2, 1]).expect("integer tensor");
            let result = block_on(ismissing_builtin(Value::Tensor(input))).expect("ismissing");
            assert!(matches!(
                result,
                Value::LogicalArray(mask)
                    if mask.shape == vec![2, 1] && mask.data == vec![0, 0]
            ));
        }
    }

    #[test]
    fn ismissing_resident_integer_uses_shape_metadata_and_returns_host_mask() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let result = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
                .expect("resident ismissing extension");
            assert!(matches!(
                result,
                Value::LogicalArray(mask)
                    if mask.shape == vec![1, 2] && mask.data == vec![0, 0]
            ));
            assert!(gpu_helpers::exact_provider_for_handle(&handle).is_some());
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn ismissing_rejects_contradictory_resident_integer_metadata() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(IntegerStorage::I8(vec![1]), vec![1, 1])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let error = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
                .expect_err("integer/logical metadata contradiction must reject");
            assert!(error.message().contains("metadata is contradictory"));
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn ismissing_matlab_mode_only_gates_explicit_resident_input() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let result = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
                    .expect("automatic residency is transparent");
                assert!(matches!(
                    result,
                    Value::LogicalArray(mask)
                        if mask.shape == vec![1, 1] && mask.data == vec![0]
                ));
            }
            runmat_accelerate_api::mark_handle_explicit(&handle);
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
                    .expect_err("explicit resident input is a compatibility-gated extension");
                assert_eq!(
                    error.identifier(),
                    ISMISSING_RESIDENT_INPUT_EXTENSION.error_identifier
                );
            }
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn anymissing_returns_false_for_every_integer_storage_class() {
        let storages = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
            IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
            IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
            IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
        ];
        for storage in storages {
            let input = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            assert_eq!(
                block_on(anymissing_builtin(Value::Tensor(input))).unwrap(),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn rmmissing_removes_rows_and_columns() {
        let value = tensor(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0], vec![3, 2]);
        let result = block_on(rmmissing_builtin(value, Vec::new())).unwrap();
        assert!(
            matches!(result, Value::Tensor(t) if t.shape == vec![2, 2] && t.materialize_f64() == vec![1.0, 2.0, 4.0, 5.0])
        );

        let value = tensor(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0], vec![3, 2]);
        let result = block_on(rmmissing_builtin(value, vec![Value::Num(2.0)])).unwrap();
        assert!(
            matches!(result, Value::Tensor(t) if t.shape == vec![3, 1] && t.materialize_f64() == vec![4.0, 5.0, 6.0])
        );
    }

    #[test]
    fn rmmissing_typed_integer_tensor_preserves_storage_and_reports_no_missing() {
        let expected = IntegerStorage::U64(vec![1, u64::MAX, 3, 4]);
        let input = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer tensor");

        let result = remove_missing_tensor(
            input,
            RemoveOptions {
                dim: RemoveDim::Rows,
            },
        )
        .unwrap();

        match result {
            (Value::Tensor(tensor), mask) => {
                assert_eq!(tensor.integer_storage(), Some(&expected));
                assert_eq!(mask.data, vec![0, 0]);
                assert_eq!(mask.shape, vec![2, 1]);
            }
            other => panic!("expected tensor and mask, got {other:?}"),
        }
    }

    #[test]
    fn rmmissing_typed_integer_vector_mask_uses_storage_len_not_mirror() {
        let expected = IntegerStorage::I16(vec![1, 2, 3]);
        let input = Tensor::new_integer(expected.clone(), vec![1, 3]).expect("integer tensor");

        let result = remove_missing_tensor(
            input,
            RemoveOptions {
                dim: RemoveDim::Rows,
            },
        )
        .unwrap();

        match result {
            (Value::Tensor(tensor), mask) => {
                assert_eq!(tensor.integer_storage(), Some(&expected));
                assert_eq!(mask.data, vec![0, 0, 0]);
                assert_eq!(mask.shape, vec![1, 3]);
            }
            other => panic!("expected tensor and mask, got {other:?}"),
        }
    }

    #[test]
    fn rmmissing_resident_integer_restores_value_and_mask_through_exact_owner() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![1, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .expect("integer tensor");
            let source = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
            runmat_accelerate_api::mark_handle_explicit(&source);
            let _outputs = crate::output_count::push_output_count(Some(2));
            let result = block_on(rmmissing_builtin(
                Value::GpuTensor(source.clone()),
                Vec::new(),
            ))
            .expect("resident rmmissing");
            let Value::OutputList(outputs) = result else {
                panic!("expected output list");
            };
            assert_eq!(outputs.len(), 2);
            for output in &outputs {
                assert!(
                    matches!(output, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
                );
            }
            assert!(
                matches!(&outputs[1], Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
            );
            let gathered_value = test_support::gather(outputs[0].clone()).expect("gather value");
            assert_eq!(
                gathered_value.integer_storage(),
                Some(&IntegerStorage::U64(vec![1, 9_007_199_254_740_993]))
            );
            let gathered_mask = test_support::gather(outputs[1].clone()).expect("gather mask");
            assert_eq!(gathered_mask.shape, vec![1, 2]);
            assert_eq!(gathered_mask.materialize_f64(), vec![0.0, 0.0]);
            assert!(gpu_helpers::exact_provider_for_handle(&source).is_some());
            for output in outputs {
                if let Value::GpuTensor(handle) = output {
                    provider.free(&handle).ok();
                }
            }
            provider.free(&source).ok();
        });
    }

    #[test]
    fn rmmissing_cell_arrays_use_cell_row_major_order() {
        let value = Value::Cell(
            CellArray::new(
                vec![
                    Value::Num(1.0),
                    Value::StringArray(
                        StringArray::new(vec![MISSING_TEXT.into()], vec![1, 1]).unwrap(),
                    ),
                    Value::Num(2.0),
                    Value::Num(3.0),
                ],
                2,
                2,
            )
            .unwrap(),
        );
        let result = block_on(rmmissing_builtin(value.clone(), Vec::new())).unwrap();
        match result {
            Value::Cell(cell) => {
                assert_eq!((cell.rows, cell.cols), (1, 2));
                assert_eq!(cell.get(0, 0).unwrap(), Value::Num(2.0));
                assert_eq!(cell.get(0, 1).unwrap(), Value::Num(3.0));
            }
            other => panic!("expected cell result, got {other:?}"),
        }

        let result = block_on(rmmissing_builtin(value, vec![Value::Num(2.0)])).unwrap();
        match result {
            Value::Cell(cell) => {
                assert_eq!((cell.rows, cell.cols), (2, 1));
                assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
                assert_eq!(cell.get(1, 0).unwrap(), Value::Num(2.0));
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn fillmissing_supports_constant_previous_and_linear() {
        let value = tensor(vec![1.0, f64::NAN, 3.0], vec![3, 1]);
        let result = block_on(fillmissing_builtin(value, vec![Value::from("linear")])).unwrap();
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 2.0, 3.0]));

        let value = tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]);
        let result = block_on(fillmissing_builtin(value, vec![Value::from("linear")])).unwrap();
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 2.0, 3.0]));

        let value = tensor(vec![1.0, f64::NAN, f64::NAN], vec![3, 1]);
        let result = block_on(fillmissing_builtin(
            value,
            vec![Value::from("constant"), Value::Num(9.0)],
        ))
        .unwrap();
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 9.0, 9.0]));
    }

    #[test]
    fn fillmissing_typed_integer_tensor_preserves_storage_and_reports_no_missing() {
        let expected = IntegerStorage::I32(vec![10, 20, 30]);
        let input = Tensor::new_integer(expected.clone(), vec![3, 1]).expect("integer tensor");

        let result = fill_missing_tensor(
            input,
            &FillOptions {
                method: FillMethod::Constant(Value::Num(0.0)),
                dim: None,
            },
        )
        .unwrap();

        match result {
            (Value::Tensor(tensor), mask) => {
                assert_eq!(tensor.integer_storage(), Some(&expected));
                assert_eq!(mask.data, vec![0, 0, 0]);
                assert_eq!(mask.shape, vec![3, 1]);
            }
            other => panic!("expected tensor and mask, got {other:?}"),
        }
    }

    #[test]
    fn fillmissing_integer_data_is_gated_and_all_classes_are_exact_noops() {
        let storages = [
            IntegerStorage::I8(vec![-1, 2]),
            IntegerStorage::I16(vec![-1, 2]),
            IntegerStorage::I32(vec![-1, 2]),
            IntegerStorage::I64(vec![-1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
        ];
        for storage in storages {
            let input = Value::Tensor(Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap());
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(fillmissing_builtin(
                    input.clone(),
                    vec![Value::from("constant"), Value::Num(0.0)],
                ))
                .expect_err("strict integer fillmissing");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:FillmissingIntegerDataExtension")
                );
            }
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let _outputs = crate::output_count::push_output_count(Some(2));
            let output = block_on(fillmissing_builtin(
                input,
                vec![Value::from("constant"), Value::Num(0.0)],
            ))
            .expect("RunMat integer fillmissing");
            let Value::OutputList(values) = output else {
                panic!("expected output list");
            };
            assert!(
                matches!(&values[0], Value::Tensor(tensor) if tensor.integer_storage() == Some(&storage))
            );
            assert!(
                matches!(&values[1], Value::LogicalArray(mask) if mask.shape == vec![2, 1] && mask.data == vec![0, 0])
            );
        }
    }

    #[test]
    fn fillmissing_integer_table_variable_and_nested_cell_are_aggregate_gated() {
        let integer = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap(),
        );
        let table = crate::builtins::table::table_from_columns(
            vec!["I".to_string()],
            vec![integer.clone()],
        )
        .unwrap();
        let nested = Value::Cell(
            CellArray::new(
                vec![Value::Cell(CellArray::new(vec![integer], 1, 1).unwrap())],
                1,
                1,
            )
            .unwrap(),
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for aggregate in [table, nested] {
            let error = block_on(fillmissing_builtin(
                aggregate,
                vec![Value::from("constant"), Value::Num(0.0)],
            ))
            .expect_err("strict aggregate integer fillmissing");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension")
            );
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn fillmissing_nested_resident_integer_is_gated_before_provider_access() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[u64::MAX]),
                    shape: &[1, 1],
                })
                .expect("integer upload");
            let nested = Value::Cell(
                CellArray::new(
                    vec![Value::Cell(
                        CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            );
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(fillmissing_builtin(
                nested,
                vec![Value::from("constant"), Value::Num(0.0)],
            ))
            .expect_err("strict nested resident integer fillmissing");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension")
            );
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn fillmissing_rejects_complex_and_sparse_arrays_instead_of_aggregate_replacement() {
        let complex = Value::ComplexTensor(
            runmat_builtins::ComplexTensor::new(vec![(f64::NAN, 0.0)], vec![1, 1]).unwrap(),
        );
        let error = fill_missing_value(
            complex,
            &FillOptions {
                method: FillMethod::Constant(Value::Num(0.0)),
                dim: None,
            },
        )
        .expect_err("complex arrays are unsupported");
        assert!(error.message().contains("complex arrays are not supported"));

        let sparse = Value::SparseTensor(
            runmat_builtins::SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap(),
        );
        let error = fill_missing_value(
            sparse,
            &FillOptions {
                method: FillMethod::Constant(Value::Num(0.0)),
                dim: None,
            },
        )
        .expect_err("sparse arrays are unsupported");
        assert!(error.message().contains("sparse arrays are not supported"));
    }

    #[test]
    fn fillmissing_nearest_uses_original_neighbors() {
        let value = tensor(vec![1.0, f64::NAN, f64::NAN, 4.0], vec![1, 4]);
        let result = block_on(fillmissing_builtin(value, vec![Value::from("nearest")])).unwrap();
        assert!(
            matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 1.0, 4.0, 4.0])
        );
    }

    #[test]
    fn fillmissing_mask_marks_only_entries_actually_filled() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let value = tensor(vec![f64::NAN, 2.0, 3.0], vec![3, 1]);
        let output = block_on(fillmissing_builtin(value, vec![Value::from("previous")])).unwrap();
        let Value::OutputList(values) = output else {
            panic!("expected output list");
        };
        assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0]));
    }

    #[test]
    fn fillmissing_rejects_unknown_options() {
        let value = tensor(vec![1.0, f64::NAN], vec![1, 2]);
        let result = block_on(fillmissing_builtin(
            value,
            vec![
                Value::from("constant"),
                Value::Num(0.0),
                Value::from("bogus"),
            ],
        ));
        assert!(result.is_err());
    }

    #[test]
    fn standardize_missing_replaces_indicators() {
        let result = block_on(standardize_missing_builtin(
            tensor(vec![-99.0, 2.0], vec![1, 2]),
            vec![Value::Num(-99.0)],
        ))
        .unwrap();
        assert!(
            matches!(result, Value::Tensor(t) if t.materialize_f64()[0].is_nan() && t.materialize_f64()[1] == 2.0)
        );
    }

    #[test]
    fn standardize_missing_reads_indicator_storage_and_does_not_nan_integer_targets() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let marker = Tensor::new_integer(IntegerStorage::I16(vec![-99]), vec![1, 1])
            .expect("integer marker");
        let expected = IntegerStorage::I16(vec![-99, 2]);
        let input = Tensor::new_integer(expected.clone(), vec![1, 2]).expect("integer input");

        let result = block_on(standardize_missing_builtin(
            Value::Tensor(input),
            vec![Value::Tensor(marker)],
        ))
        .unwrap();

        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.integer_storage(), Some(&expected));
                assert_eq!(tensor.materialize_f64(), vec![-99.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn standardize_missing_integer_data_is_mode_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![-99, 2]), vec![1, 2]).unwrap(),
        );
        let error = block_on(standardize_missing_builtin(input, vec![Value::Num(-99.0)]))
            .expect_err("MATLAB-compatible mode must reject direct integer data");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:StandardizeMissingIntegerDataExtension")
        );
    }

    #[test]
    fn standardize_missing_documented_integer_indicator_needs_no_extension() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let marker =
            Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![-99]), vec![1, 1]).unwrap());
        let result = block_on(standardize_missing_builtin(
            tensor(vec![-99.0, 2.0], vec![1, 2]),
            vec![marker],
        ))
        .expect("documented integer indicator");
        assert!(
            matches!(result, Value::Tensor(t) if t.materialize_f64()[0].is_nan() && t.materialize_f64()[1] == 2.0)
        );
    }

    #[test]
    fn standardize_missing_explicit_gpu_indicator_is_gated_before_provider_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_451_002,
        };
        runmat_accelerate_api::mark_handle_explicit(&handle);
        let error = block_on(standardize_missing_builtin(
            tensor(vec![-99.0, 2.0], vec![1, 2]),
            vec![Value::GpuTensor(handle)],
        ))
        .expect_err("explicit GPU indicator must be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:StandardizeMissingExplicitGpuIndicatorExtension")
        );
    }

    #[test]
    fn nanmean_alias_uses_omitnan() {
        let result = block_on(nanmean_builtin(
            tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]),
            vec![Value::from("all")],
        ))
        .unwrap();
        assert!(matches!(result, Value::Num(n) if (n - 2.0).abs() < 1e-12));
    }

    #[test]
    fn legacy_nan_reductions_gate_typed_integer_data_by_compatibility_mode() {
        let cases: [(fn(Value, Vec<Value>) -> BuiltinResult<Value>, &str, &str); 4] = [
            (
                |value, rest| block_on(nanmean_builtin(value, rest)),
                "nanmean",
                "RunMat:compatibility:NanmeanTypedIntegerInputExtension",
            ),
            (
                |value, rest| block_on(nansum_builtin(value, rest)),
                "nansum",
                "RunMat:compatibility:NansumTypedIntegerInputExtension",
            ),
            (
                |value, rest| block_on(nanmin_builtin(value, rest)),
                "nanmin",
                "RunMat:compatibility:NanminTypedIntegerInputExtension",
            ),
            (
                |value, rest| block_on(nanmedian_builtin(value, rest)),
                "nanmedian",
                "RunMat:compatibility:NanmedianTypedIntegerInputExtension",
            ),
        ];

        for (invoke, name, identifier) in cases {
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = invoke(Value::Int(IntValue::U16(7)), Vec::new()).unwrap_err();
                assert_eq!(error.identifier(), Some(identifier), "{name}");
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                invoke(Value::Int(IntValue::U16(7)), Vec::new())
                    .unwrap_or_else(|error| panic!("{name} RunMat typed-integer input: {error}"));
            }
        }
    }

    #[test]
    fn legacy_nan_reductions_gate_typed_integer_dimensions() {
        let floating = || tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let dimension = || Value::Int(IntValue::U8(2));
        let placeholder = || {
            Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty placeholder"))
        };

        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(nanmean_builtin(floating(), vec![dimension()]))
            .expect_err("strict nanmean typed-integer dimension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension")
        );
        let error = block_on(nansum_builtin(floating(), vec![dimension()]))
            .expect_err("strict nansum typed-integer dimension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NansumTypedIntegerInputExtension")
        );
        let error = block_on(nanmedian_builtin(floating(), vec![dimension()]))
            .expect_err("strict nanmedian typed-integer dimension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanmedianTypedIntegerInputExtension")
        );
        let error = block_on(nanmin_builtin(floating(), vec![placeholder(), dimension()]))
            .expect_err("strict nanmin typed-integer dimension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanminTypedIntegerInputExtension")
        );
    }

    #[test]
    fn legacy_nan_std_var_reject_integer_data_and_gate_integer_controls() {
        for enabled in [false, true] {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(enabled);
            let error = block_on(nanstd_builtin(
                Value::Int(IntValue::I16(7)),
                vec![Value::Int(IntValue::U8(1))],
            ))
            .expect_err("nanstd integer data");
            assert!(error.message().contains("integer data inputs"));
            let error = block_on(nanvar_builtin(
                Value::Int(IntValue::I16(7)),
                vec![Value::Int(IntValue::U8(1))],
            ))
            .expect_err("nanvar integer data");
            assert!(error.message().contains("integer data inputs"));
        }

        let floating = || tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(nanstd_builtin(
                floating(),
                vec![Value::Int(IntValue::U8(1))],
            ))
            .expect_err("nanstd strict typed-integer control");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:NanstdTypedIntegerControlExtension")
            );
            let error = block_on(nanvar_builtin(
                floating(),
                vec![Value::Int(IntValue::U8(1))],
            ))
            .expect_err("nanvar strict typed-integer control");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:NanvarTypedIntegerControlExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(nanstd_builtin(
                floating(),
                vec![Value::Int(IntValue::U8(1))],
            ))
            .expect("nanstd RunMat typed-integer control");
            block_on(nanvar_builtin(
                floating(),
                vec![Value::Int(IntValue::U8(1))],
            ))
            .expect("nanvar RunMat typed-integer control");
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn legacy_nan_integer_policy_inspects_resident_dtype_before_dispatch() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1, u64::MAX]),
                    shape: &[2, 1],
                })
                .expect("integer upload");

            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(nanmean_builtin(
                    Value::GpuTensor(handle.clone()),
                    Vec::new(),
                ))
                .expect_err("strict resident nanmean");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension")
                );

                let error = block_on(nanstd_builtin(Value::GpuTensor(handle.clone()), Vec::new()))
                    .expect_err("resident nanstd integer data");
                assert!(error.message().contains("integer data inputs"));
            }

            provider.free(&handle).ok();
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn legacy_nan_integer_extensions_preserve_declared_residency() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let cases: [(
                &str,
                fn(Value, Vec<Value>) -> BuiltinResult<Value>,
                Vec<f64>,
            ); 2] = [
                (
                    "nanmean",
                    |value, rest| block_on(nanmean_builtin(value, rest)),
                    vec![2.0, 6.0],
                ),
                (
                    "nansum",
                    |value, rest| block_on(nansum_builtin(value, rest)),
                    vec![4.0, 12.0],
                ),
            ];
            for (name, invoke, expected) in cases {
                let handle = provider
                    .upload_integer(&HostIntegerTensorView {
                        data: HostIntegerDataView::U64(&[1, 3, 5, 7]),
                        shape: &[2, 2],
                    })
                    .expect("integer upload");
                let result = invoke(Value::GpuTensor(handle), Vec::new())
                    .expect("resident integer extension");
                assert!(
                    matches!(result, Value::GpuTensor(_)),
                    "{name} returned {result:?}"
                );
                let gathered = test_support::gather(result).expect("gather result");
                assert_eq!(gathered.materialize_f64(), expected);
            }

            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1, 3, 5, 7]),
                    shape: &[2, 2],
                })
                .expect("integer upload");
            let result = block_on(nanmedian_builtin(Value::GpuTensor(handle), Vec::new()))
                .expect("resident integer nanmedian");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather median");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![2, 6]))
            );
        });
    }

    #[test]
    fn nanmin_supports_pairwise_form() {
        let result = block_on(nanmin_builtin(
            tensor(vec![f64::NAN, 4.0, 3.0], vec![1, 3]),
            vec![tensor(vec![2.0, f64::NAN, 5.0], vec![1, 3])],
        ))
        .unwrap();
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![2.0, 4.0, 3.0]));
    }

    #[test]
    fn nanmin_pairwise_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let left = Tensor::new_integer(IntegerStorage::U16(vec![9, 4, 3]), vec![1, 3]).unwrap();
        let right = tensor(vec![2.0, f64::NAN, 5.0], vec![1, 3]);

        let result = block_on(nanmin_builtin(Value::Tensor(left), vec![right])).unwrap();

        assert!(
            matches!(result, Value::Tensor(tensor) if tensor.materialize_f64() == vec![2.0, 4.0, 3.0])
        );

        let left = tensor(vec![9.0, 4.0, 3.0], vec![1, 3]);
        let right = Tensor::new_integer(IntegerStorage::U8(vec![5]), vec![1, 1]).unwrap();

        let result = block_on(nanmin_builtin(left, vec![Value::Tensor(right)])).unwrap();

        assert!(
            matches!(result, Value::Tensor(tensor) if tensor.materialize_f64() == vec![5.0, 4.0, 3.0])
        );
    }

    #[test]
    fn nanmin_pairwise_preserves_exact_wide_same_class_integers() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let left = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("left uint64");
        let right = Tensor::new_integer(
            IntegerStorage::U64(vec![1_u64 << 53, u64::MAX - 1]),
            vec![1, 2],
        )
        .expect("right uint64");

        let result = block_on(nanmin_builtin(
            Value::Tensor(left),
            vec![Value::Tensor(right)],
        ))
        .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected uint64 tensor");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 53, u64::MAX - 1]))
        );
    }

    #[test]
    fn movmad_computes_centered_median_absolute_deviation() {
        let result = block_on(movmad_builtin(
            tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]),
            Value::Num(3.0),
            Vec::new(),
        ))
        .unwrap();
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64()[2] == 2.0));
    }

    #[test]
    fn movmad_reads_typed_integer_storage_and_returns_double_output() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1])
            .expect("integer movmad input");

        let result = block_on(movmad_builtin(
            Value::Tensor(input),
            Value::Num(3.0),
            Vec::new(),
        ))
        .unwrap();

        match result {
            Value::Tensor(tensor) => {
                assert!(tensor.integer_storage().is_none());
                assert_eq!(tensor.materialize_f64(), vec![0.5, 1.0, 2.0, 1.0, 0.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn movmad_accepts_all_integer_classes_into_double_output() {
        let storages = vec![
            IntegerStorage::I8(vec![1, 2, 5]),
            IntegerStorage::I16(vec![1, 2, 5]),
            IntegerStorage::I32(vec![1, 2, 5]),
            IntegerStorage::I64(vec![1, 2, 5]),
            IntegerStorage::U8(vec![1, 2, 5]),
            IntegerStorage::U16(vec![1, 2, 5]),
            IntegerStorage::U32(vec![1, 2, 5]),
            IntegerStorage::U64(vec![1, 2, 5]),
        ];
        for storage in storages {
            let input = Tensor::new_integer(storage, vec![3, 1]).unwrap();
            let result = block_on(movmad_builtin(
                Value::Tensor(input),
                Value::Num(3.0),
                Vec::new(),
            ))
            .unwrap();
            assert!(
                matches!(result, Value::Tensor(tensor) if tensor.integer_storage().is_none() && tensor.materialize_f64() == vec![0.5, 1.0, 1.5])
            );
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn movmad_gpu_large_window_follows_compatibility_mode() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new((1..=32).map(f64::from).collect(), vec![32, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(movmad_builtin(
                    Value::GpuTensor(handle.clone()),
                    Value::Num(32.0),
                    Vec::new(),
                ))
                .unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:MovmadGpuLargeWindowExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let result = block_on(movmad_builtin(
                    Value::GpuTensor(handle.clone()),
                    Value::Num(32.0),
                    Vec::new(),
                ))
                .unwrap();
                assert!(matches!(result, Value::GpuTensor(_)));
            }
            let _ = provider.free(&handle);
        });
    }
}
