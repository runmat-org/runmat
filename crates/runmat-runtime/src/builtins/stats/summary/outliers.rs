//! Outlier detection and filling helpers.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, IntegerStorage, LogicalArray, NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ISOUTLIER_NAME: &str = "isoutlier";
const FILLOUTLIERS_NAME: &str = "filloutliers";
const MAD_SCALE: f64 = 1.4826;
const MAX_EXACT_INTEGER_F64: u128 = 1_u128 << 53;

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric array.",
};

const PARAM_FILL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fillmethod",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Replacement method or constant value.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Detection method, dimension, and name-value options.",
};

const OUTPUT_MASK: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "TF",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask of detected outliers.",
};

const OUTPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array with outliers filled.",
};

const OUTPUT_LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Lower threshold.",
};

const OUTPUT_UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "U",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Upper threshold.",
};

const OUTPUT_CENTER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Center value used by the detection method.",
};

const INPUT_A: [BuiltinParamDescriptor; 1] = [PARAM_A];
const INPUT_A_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_OPTIONS];
const INPUT_A_FILL: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_FILL];
const INPUT_A_FILL_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_FILL, PARAM_OPTIONS];
const OUT_MASK: [BuiltinParamDescriptor; 1] = [OUTPUT_MASK];
const OUT_MASK_THRESHOLDS: [BuiltinParamDescriptor; 4] =
    [OUTPUT_MASK, OUTPUT_LOWER, OUTPUT_UPPER, OUTPUT_CENTER];
const OUT_VALUE: [BuiltinParamDescriptor; 1] = [OUTPUT_VALUE];
const OUT_VALUE_MASK: [BuiltinParamDescriptor; 2] = [OUTPUT_VALUE, OUTPUT_MASK];
const OUT_VALUE_THRESHOLDS: [BuiltinParamDescriptor; 5] = [
    OUTPUT_VALUE,
    OUTPUT_MASK,
    OUTPUT_LOWER,
    OUTPUT_UPPER,
    OUTPUT_CENTER,
];

const ISOUTLIER_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "TF = isoutlier(A)",
        inputs: &INPUT_A,
        outputs: &OUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "TF = isoutlier(A, method)",
        inputs: &INPUT_A_OPTIONS,
        outputs: &OUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "TF = isoutlier(A, ___, dim)",
        inputs: &INPUT_A_OPTIONS,
        outputs: &OUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[TF, L, U, C] = isoutlier(___)",
        inputs: &INPUT_A_OPTIONS,
        outputs: &OUT_MASK_THRESHOLDS,
    },
];

const FILLOUTLIERS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "B = filloutliers(A, fillmethod)",
        inputs: &INPUT_A_FILL,
        outputs: &OUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "B = filloutliers(A, fillmethod, findmethod)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "B = filloutliers(A, ___, dim)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, TF] = filloutliers(___)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[B, TF, L, U, C] = filloutliers(___)",
        inputs: &INPUT_A_FILL_OPTIONS,
        outputs: &OUT_VALUE_THRESHOLDS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.INVALID_ARGUMENT",
    identifier: Some("RunMat:outliers:InvalidArgument"),
    when: "Inputs, methods, dimensions, or name-value options are malformed.",
    message: "outlier builtin: invalid argument",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.UNSUPPORTED",
    identifier: Some("RunMat:outliers:Unsupported"),
    when: "A table/timetable or advanced smoothing/interpolation form is requested.",
    message: "outlier builtin: unsupported form",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.INTERNAL",
    identifier: Some("RunMat:outliers:Internal"),
    when: "Internal tensor or logical array materialization fails.",
    message: "outlier builtin: internal error",
};

const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OUTLIERS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:outliers:TooManyOutputs"),
    when: "More outputs are requested than the builtin can return.",
    message: "outlier builtin: too many outputs",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_UNSUPPORTED,
    ERROR_INTERNAL,
    ERROR_TOO_MANY_OUTPUTS,
];

pub const ISOUTLIER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISOUTLIER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ISOUTLIER_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "isoutlier-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "isoutlier with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IsoutlierIntegerDataExtension"),
};
const ISOUTLIER_INTEGER_DIMENSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-integer-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isoutlier with a typed-integer dimension is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierIntegerDimensionExtension"),
    };
const ISOUTLIER_INTEGER_WINDOW_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "isoutlier-integer-window",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "isoutlier with a typed-integer moving window is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IsoutlierIntegerWindowExtension"),
};
const ISOUTLIER_INTEGER_THRESHOLD_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-integer-threshold",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isoutlier with typed-integer percentile or ThresholdFactor values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierIntegerThresholdExtension"),
    };
const ISOUTLIER_GPU_MOVMEDIAN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "isoutlier-gpu-movmedian",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "an explicit GPU isoutlier call using movmedian is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IsoutlierGpuMovmedianExtension"),
};
const ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-gpu-sample-points",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "an explicit GPU isoutlier call using SamplePoints is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierGpuSamplePointsExtension"),
    };
const ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isoutlier-gpu-data-variables",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "an explicit GPU isoutlier call using DataVariables is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsoutlierGpuDataVariablesExtension"),
    };

pub const ISOUTLIER_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    ISOUTLIER_INTEGER_DATA_EXTENSION,
    ISOUTLIER_INTEGER_DIMENSION_EXTENSION,
    ISOUTLIER_INTEGER_WINDOW_EXTENSION,
    ISOUTLIER_INTEGER_THRESHOLD_EXTENSION,
    ISOUTLIER_GPU_MOVMEDIAN_EXTENSION,
    ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION,
    ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION,
];

const ISOUTLIER_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric data classes are single and double. Each RunMat-only typed value must be exactly representable at the binary64 statistics boundary.",
    }];
const ISOUTLIER_INTEGER_STRUCTURAL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation specifies a positive integer value domain but does not enumerate native integer storage classes, so typed storage remains conservatively RunMat-only.",
    },
    BuiltinIntegerInputCapability {
        name: "window",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Count-form moving windows are decoded exactly; duration windows are a separate, currently unsupported surface.",
    },
];
const ISOUTLIER_INTEGER_THRESHOLD_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "threshold",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed percentile bounds are not publicly class-enumerated and must be exactly representable at the floating percentile boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "ThresholdFactor",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed threshold factors are not publicly class-enumerated and must be exactly representable at the floating statistics boundary.",
    },
];

pub const ISOUTLIER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[TF, L, U, C] = isoutlier(integer_A, ...)",
        inputs: &ISOUTLIER_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "TF is logical; threshold outputs are floating. Compatibility and exactness are checked before provider access, and the current CPU fallback may restore supported outputs to the source provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "isoutlier(A, ..., integer_dim/window)",
        inputs: &ISOUTLIER_INTEGER_STRUCTURAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "These ambiguous public storage-class forms stay independently RunMat-gated rather than being overclaimed as documented native-integer support.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "isoutlier(A, percentiles, integer_threshold / ThresholdFactor=integer_factor)",
        inputs: &ISOUTLIER_INTEGER_THRESHOLD_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Ambiguous typed storage is admitted only in RunMat mode and only when every value is exact in binary64.",
    },
];

pub const FILLOUTLIERS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILLOUTLIERS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const FILLOUTLIERS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "filloutliers with typed-integer input data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersIntegerDataExtension"),
    };
const FILLOUTLIERS_INTEGER_FILL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-integer-fill-scalar",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "filloutliers with a typed-integer constant fill scalar is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersIntegerFillExtension"),
    };
const FILLOUTLIERS_NUMERIC_MASK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-numeric-outlier-locations",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "filloutliers with numeric rather than logical OutlierLocations is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersNumericMaskExtension"),
    };
const FILLOUTLIERS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "filloutliers-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "filloutliers with a direct provider-resident input or argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FilloutliersResidentInputExtension"),
    };
pub const FILLOUTLIERS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FILLOUTLIERS_INTEGER_DATA_EXTENSION,
    FILLOUTLIERS_INTEGER_FILL_EXTENSION,
    FILLOUTLIERS_NUMERIC_MASK_EXTENSION,
    FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
];

const FILLOUTLIERS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight classes are admitted only when the exact input span is at most 2^53. Wider spans are rejected before statistics rather than collapsing distinct observations at the double boundary.",
    }];
const FILLOUTLIERS_INTEGER_FILL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fillmethod",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A typed-integer scalar constant is decoded from authoritative storage and converted once to the output computation domain.",
    }];
const FILLOUTLIERS_INTEGER_MASK_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "OutlierLocations",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented masks are logical; numeric masks are accepted only in RunMat mode and must exactly match A's shape.",
    }];
pub const FILLOUTLIERS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[B, TF, L, U, C] = filloutliers(integer_A, ...)",
        inputs: &FILLOUTLIERS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "For admitted inputs, exact anchored integer differences are representable in binary64. Inputs spanning more than 2^53 are rejected; B, L, U, and C deliberately cross to double, while TF is logical.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "filloutliers(A, integer_fill_scalar, ...)",
        inputs: &FILLOUTLIERS_INTEGER_FILL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The typed-integer constant-fill role is independently mode-gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "filloutliers(A, ..., OutlierLocations=integer_mask)",
        inputs: &FILLOUTLIERS_INTEGER_MASK_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Zero is false and nonzero is true after the independently gated numeric-mask extension is admitted.",
    },
];

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn logical_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Logical { shape: None }
}

fn outlier_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.into()))
        .with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    outlier_error(builtin, &ERROR_INVALID_ARGUMENT, detail)
}

fn unsupported(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    outlier_error(builtin, &ERROR_UNSUPPORTED, detail)
}

fn internal_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    outlier_error(builtin, &ERROR_INTERNAL, detail)
}

fn too_many_outputs(builtin: &'static str, max_outputs: usize) -> RuntimeError {
    outlier_error(
        builtin,
        &ERROR_TOO_MANY_OUTPUTS,
        format!("{builtin}: requested more than {max_outputs} outputs"),
    )
}

#[derive(Clone, Debug)]
enum DetectionMethod {
    Median,
    Mean,
    Quartiles,
    Percentiles(f64, f64),
    MovingMedian(usize),
    MovingMean(usize),
}

#[derive(Clone, Debug)]
struct DetectionOptions {
    method: DetectionMethod,
    method_was_specified: bool,
    dim: Option<usize>,
    threshold_factor: Option<f64>,
    forced_mask: Option<(Vec<u8>, Vec<usize>)>,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            method: DetectionMethod::Median,
            method_was_specified: false,
            dim: None,
            threshold_factor: None,
            forced_mask: None,
        }
    }
}

#[derive(Clone, Debug)]
enum FillMethod {
    Center,
    Clip,
    Constant(f64),
    Previous,
    Next,
    Nearest,
    Linear,
}

#[derive(Clone, Debug)]
struct DetectionResult {
    mask: Vec<u8>,
    mask_shape: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    center: Vec<f64>,
    lower_full: Vec<f64>,
    upper_full: Vec<f64>,
    center_full: Vec<f64>,
    threshold_shape: Vec<usize>,
}

#[runtime_builtin(
    name = "isoutlier",
    category = "stats/summary",
    summary = "Detect outliers in numeric arrays.",
    keywords = "isoutlier,outlier,median,quartiles,mean,statistics",
    accel = "cpu",
    type_resolver(logical_type),
    descriptor(crate::builtins::stats::summary::outliers::ISOUTLIER_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::outliers::ISOUTLIER_EXTENSIONS),
    integer_capabilities(
        crate::builtins::stats::summary::outliers::ISOUTLIER_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::stats::summary::outliers"
)]
pub(crate) async fn isoutlier_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_isoutlier_extensions(&value, &rest)?;
    ensure_isoutlier_exact_boundaries(&value, &rest).await?;
    let output_source = isoutlier_gpu_source(&value);
    let tensor = value_to_tensor(ISOUTLIER_NAME, value).await?;
    let options = parse_detection_options(ISOUTLIER_NAME, rest).await?;
    let result = detect_outliers(&tensor, &options, ISOUTLIER_NAME)?;
    let value = outlier_outputs(ISOUTLIER_NAME, None, result)?;
    restore_isoutlier_value(value, output_source.as_ref())
}

fn isoutlier_gpu_source(value: &Value) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    Some(handle.clone())
}

fn restore_isoutlier_value(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    match value {
        Value::Tensor(tensor) => restore_isoutlier_array(Value::Tensor(tensor), source),
        Value::LogicalArray(logical) => {
            restore_isoutlier_array(Value::LogicalArray(logical), source)
        }
        Value::OutputList(values) => values
            .into_iter()
            .map(|value| restore_isoutlier_value(value, Some(source)))
            .collect::<BuiltinResult<Vec<_>>>()
            .map(Value::OutputList),
        other => Ok(other),
    }
}

fn restore_isoutlier_array(
    value: Value,
    source: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let restored = crate::builtins::common::gpu_helpers::restore_class_preserving_value(
        source,
        value,
        ISOUTLIER_NAME,
    )?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(internal_error(
            ISOUTLIER_NAME,
            "isoutlier: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

fn explicit_isoutlier_gpu(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
}

fn ensure_isoutlier_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ISOUTLIER_INTEGER_DATA_EXTENSION,
            ISOUTLIER_NAME,
        )?;
    }
    let explicit_gpu = explicit_isoutlier_gpu(value) || rest.iter().any(explicit_isoutlier_gpu);
    let mut idx = 0;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            let keyword = keyword.to_ascii_lowercase();
            match keyword.as_str() {
                "movmedian" | "movmean" => {
                    if keyword == "movmedian" && explicit_gpu {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_GPU_MOVMEDIAN_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    if rest.get(idx + 1).is_some_and(is_typed_integer_value) {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_INTEGER_WINDOW_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                "samplepoints" => {
                    if explicit_gpu {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                "data variables" | "datavariables" => {
                    if explicit_gpu {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                "percentiles" | "thresholdfactor" => {
                    if rest.get(idx + 1).is_some_and(is_typed_integer_value) {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_INTEGER_THRESHOLD_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        } else if is_typed_integer_value(&rest[idx]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ISOUTLIER_INTEGER_DIMENSION_EXTENSION,
                ISOUTLIER_NAME,
            )?;
        }
        idx += 1;
    }
    Ok(())
}

async fn ensure_isoutlier_exact_boundaries(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    ensure_isoutlier_exact_f64(value, "input data").await?;
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if keyword_of(&rest[idx]).is_some_and(|keyword| {
            matches!(
                keyword.to_ascii_lowercase().as_str(),
                "percentiles" | "thresholdfactor"
            )
        }) {
            ensure_isoutlier_exact_f64(&rest[idx + 1], "threshold value").await?;
        }
        idx += 1;
    }
    Ok(())
}

async fn ensure_isoutlier_exact_f64(value: &Value, role: &str) -> BuiltinResult<()> {
    if is_typed_integer_value(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(invalid_argument(
            ISOUTLIER_NAME,
            format!("typed-integer {role} must be exactly representable as double"),
        ));
    }
    Ok(())
}

#[runtime_builtin(
    name = "filloutliers",
    category = "stats/summary",
    summary = "Fill outliers in numeric arrays.",
    keywords = "filloutliers,outlier,fill,median,quartiles,mean,statistics",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::stats::summary::outliers::FILLOUTLIERS_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::outliers::FILLOUTLIERS_EXTENSIONS),
    integer_capabilities(
        crate::builtins::stats::summary::outliers::FILLOUTLIERS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::stats::summary::outliers"
)]
pub(crate) async fn filloutliers_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_filloutliers_extensions(&value, &rest)?;
    let tensor = value_to_tensor(FILLOUTLIERS_NAME, value).await?;
    let mut rest = gather_values(rest).await?;
    if rest.is_empty() {
        return Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: fill method is required",
        ));
    }
    let fill_method = parse_fill_method(&mut rest)?;
    let options = parse_detection_options(FILLOUTLIERS_NAME, rest).await?;
    let mut result = detect_outliers(&tensor, &options, FILLOUTLIERS_NAME)?;
    let (filled, filled_mask) = fill_outlier_tensor(&tensor, &result, &options, &fill_method)?;
    result.mask = filled_mask;
    outlier_outputs(FILLOUTLIERS_NAME, Some(filled), result)
}

fn ensure_filloutliers_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_INTEGER_DATA_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    if rest.first().is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_INTEGER_FILL_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if keyword_of(&rest[idx]).is_some_and(|name| name.eq_ignore_ascii_case("outlierlocations"))
            && is_numeric_mask_value(&rest[idx + 1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FILLOUTLIERS_NUMERIC_MASK_EXTENSION,
                FILLOUTLIERS_NAME,
            )?;
        }
        idx += 1;
    }
    if rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    Ok(())
}

fn is_numeric_mask_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_if_needed_async(&value).await?);
    }
    Ok(out)
}

async fn value_to_tensor(builtin: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        invalid_argument(builtin, format!("{builtin}: failed to gather input: {err}"))
    })?;
    tensor::value_into_tensor_for(builtin, value)
        .map_err(|err| invalid_argument(builtin, format!("{builtin}: {err}")))
}

async fn parse_detection_options(
    builtin: &'static str,
    rest: Vec<Value>,
) -> BuiltinResult<DetectionOptions> {
    let rest = gather_values(rest).await?;
    let mut options = DetectionOptions::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let arg = &rest[idx];
        if let Some(keyword) = keyword_of(arg) {
            match keyword.to_ascii_lowercase().as_str() {
                "median" => {
                    options.method = DetectionMethod::Median;
                    options.method_was_specified = true;
                }
                "mean" => {
                    options.method = DetectionMethod::Mean;
                    options.method_was_specified = true;
                }
                "quartiles" => {
                    options.method = DetectionMethod::Quartiles;
                    options.method_was_specified = true;
                }
                "percentiles" => {
                    options.method_was_specified = true;
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles requires [lower upper]",
                        ));
                    }
                    let bounds = numeric_vector(builtin, &rest[idx])?;
                    if bounds.len() != 2 {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles must be a two-element vector",
                        ));
                    }
                    if !(0.0..=100.0).contains(&bounds[0])
                        || !(0.0..=100.0).contains(&bounds[1])
                        || bounds[0] >= bounds[1]
                    {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles must satisfy 0 <= p1 < p2 <= 100",
                        ));
                    }
                    options.method =
                        DetectionMethod::Percentiles(bounds[0] / 100.0, bounds[1] / 100.0);
                }
                "movmedian" | "movmean" => {
                    options.method_was_specified = true;
                    let is_median = keyword.eq_ignore_ascii_case("movmedian");
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(builtin, "moving method requires a window"));
                    }
                    let window = scalar_usize(builtin, &rest[idx], "window")?;
                    if window == 0 {
                        return Err(invalid_argument(builtin, "window must be positive"));
                    }
                    options.method = if is_median {
                        DetectionMethod::MovingMedian(window)
                    } else {
                        DetectionMethod::MovingMean(window)
                    };
                }
                "thresholdfactor" => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "ThresholdFactor requires a numeric scalar",
                        ));
                    }
                    let factor = scalar_number(&rest[idx]).ok_or_else(|| {
                        invalid_argument(builtin, "ThresholdFactor must be a numeric scalar")
                    })?;
                    if !(factor.is_finite() && factor >= 0.0) {
                        return Err(invalid_argument(
                            builtin,
                            "ThresholdFactor must be a finite nonnegative scalar",
                        ));
                    }
                    options.threshold_factor = Some(factor);
                }
                "outlierlocations" if builtin == FILLOUTLIERS_NAME => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "OutlierLocations requires a logical mask value",
                        ));
                    }
                    options.forced_mask = Some(logical_mask(builtin, &rest[idx])?);
                }
                "samplepoints" | "data variables" | "datavariables" | "outlierlocations" => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            format!("{keyword} requires a value"),
                        ));
                    }
                    return Err(unsupported(
                        builtin,
                        format!(
                            "{keyword} is not supported for numeric-array outlier detection yet"
                        ),
                    ));
                }
                other => {
                    return Err(invalid_argument(
                        builtin,
                        format!("unsupported option or method '{other}'"),
                    ));
                }
            }
        } else {
            options.dim = Some(parse_dim(builtin, arg)?);
        }
        idx += 1;
    }
    if matches!(options.method, DetectionMethod::Percentiles(_, _))
        && options.threshold_factor.is_some()
    {
        return Err(invalid_argument(
            builtin,
            "ThresholdFactor is not supported with the percentiles method",
        ));
    }
    if options.forced_mask.is_some() && options.method_was_specified {
        return Err(invalid_argument(
            builtin,
            "OutlierLocations cannot be combined with a specified find method",
        ));
    }
    Ok(options)
}

fn parse_fill_method(rest: &mut Vec<Value>) -> BuiltinResult<FillMethod> {
    let value = rest.remove(0);
    if let Some(number) = scalar_number(&value) {
        return Ok(FillMethod::Constant(number));
    }
    let Some(keyword) = keyword_of(&value) else {
        return Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: fill method must be text or numeric scalar",
        ));
    };
    match keyword.to_ascii_lowercase().as_str() {
        "center" => Ok(FillMethod::Center),
        "clip" => Ok(FillMethod::Clip),
        "previous" => Ok(FillMethod::Previous),
        "next" => Ok(FillMethod::Next),
        "nearest" => Ok(FillMethod::Nearest),
        "linear" => Ok(FillMethod::Linear),
        "constant" => Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: use a numeric scalar fill method for constant replacement",
        )),
        "spline" | "pchip" | "makima" => Err(unsupported(
            FILLOUTLIERS_NAME,
            format!("filloutliers: interpolation method '{keyword}' is not supported yet"),
        )),
        other => Err(invalid_argument(
            FILLOUTLIERS_NAME,
            format!("filloutliers: unsupported fill method '{other}'"),
        )),
    }
}

fn detect_outliers(
    input: &Tensor,
    options: &DetectionOptions,
    builtin: &'static str,
) -> BuiltinResult<DetectionResult> {
    let (input_values, integer_offset) = statistical_values(input, builtin)?;
    if let Some((_, mask_shape)) = &options.forced_mask {
        if mask_shape != &input.shape {
            return Err(invalid_argument(
                builtin,
                "OutlierLocations must have exactly the same shape as the input",
            ));
        }
    }
    let shape = tensor::default_shape_for(&input.shape, input_values.len());
    let dim = options.dim.unwrap_or_else(|| first_non_singleton(&shape));
    let mut result = if dim == 0 {
        detect_all(input, &input_values, options)?
    } else {
        let axis = dim - 1;
        let rank = shape.len().max(axis + 1);
        let mut padded_shape = shape.clone();
        padded_shape.resize(rank, 1);
        if matches!(
            options.method,
            DetectionMethod::MovingMedian(_) | DetectionMethod::MovingMean(_)
        ) {
            detect_moving(input, &input_values, &padded_shape, axis, options)?
        } else {
            detect_by_slice(input, &input_values, &padded_shape, axis, options)?
        }
    };
    if let Some((mask, _)) = &options.forced_mask {
        result.mask.clone_from(mask);
    }
    if integer_offset != 0.0 {
        for values in [
            &mut result.lower,
            &mut result.upper,
            &mut result.center,
            &mut result.lower_full,
            &mut result.upper_full,
            &mut result.center_full,
        ] {
            for value in values {
                *value += integer_offset;
            }
        }
    }
    Ok(result)
}

fn statistical_values(input: &Tensor, builtin: &'static str) -> BuiltinResult<(Vec<f64>, f64)> {
    let Some(storage) = input.integer_storage() else {
        return Ok((tensor::tensor_values_f64(input), 0.0));
    };
    macro_rules! signed_offsets {
        ($values:expr) => {{
            let minimum = $values.iter().copied().min().unwrap_or(0) as i128;
            let maximum = $values.iter().copied().max().unwrap_or(0) as i128;
            let span = (maximum - minimum) as u128;
            if span > MAX_EXACT_INTEGER_F64 {
                return Err(invalid_argument(
                    builtin,
                    "typed-integer input span exceeds 2^53 and cannot enter the lossless outlier-statistics input domain",
                ));
            }
            Ok((
                $values
                    .iter()
                    .map(|value| ((*value as i128) - minimum) as f64)
                    .collect(),
                minimum as f64,
            ))
        }};
    }
    macro_rules! unsigned_offsets {
        ($values:expr) => {{
            let minimum = $values.iter().copied().min().unwrap_or(0) as u128;
            let maximum = $values.iter().copied().max().unwrap_or(0) as u128;
            let span = maximum - minimum;
            if span > MAX_EXACT_INTEGER_F64 {
                return Err(invalid_argument(
                    builtin,
                    "typed-integer input span exceeds 2^53 and cannot enter the lossless outlier-statistics input domain",
                ));
            }
            Ok((
                $values
                    .iter()
                    .map(|value| ((*value as u128) - minimum) as f64)
                    .collect(),
                minimum as f64,
            ))
        }};
    }
    match storage {
        IntegerStorage::I8(values) => signed_offsets!(values),
        IntegerStorage::I16(values) => signed_offsets!(values),
        IntegerStorage::I32(values) => signed_offsets!(values),
        IntegerStorage::I64(values) => signed_offsets!(values),
        IntegerStorage::U8(values) => unsigned_offsets!(values),
        IntegerStorage::U16(values) => unsigned_offsets!(values),
        IntegerStorage::U32(values) => unsigned_offsets!(values),
        IntegerStorage::U64(values) => unsigned_offsets!(values),
    }
}

fn detect_all(
    input: &Tensor,
    input_values: &[f64],
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    let stats = slice_stats(input_values, &options.method, threshold_factor(options))?;
    let mask = input_values
        .iter()
        .map(|value| u8::from(is_outlier_value(*value, stats.lower, stats.upper)))
        .collect();
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape.clone(),
        lower: vec![stats.lower],
        upper: vec![stats.upper],
        center: vec![stats.center],
        lower_full: vec![stats.lower; input_values.len()],
        upper_full: vec![stats.upper; input_values.len()],
        center_full: vec![stats.center; input_values.len()],
        threshold_shape: vec![1, 1],
    })
}

fn detect_by_slice(
    input: &Tensor,
    input_values: &[f64],
    shape: &[usize],
    axis: usize,
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    let axis_len = shape[axis];
    let pre: usize = shape[..axis].iter().product();
    let post: usize = shape[axis + 1..].iter().product();
    let mut threshold_shape = shape.to_vec();
    threshold_shape[axis] = 1;
    let threshold_len = tensor::element_count(&threshold_shape);
    let mut mask = vec![0u8; input_values.len()];
    let mut lower = vec![f64::NAN; threshold_len];
    let mut upper = vec![f64::NAN; threshold_len];
    let mut center = vec![f64::NAN; threshold_len];
    let mut lower_full = vec![f64::NAN; input_values.len()];
    let mut upper_full = vec![f64::NAN; input_values.len()];
    let mut center_full = vec![f64::NAN; input_values.len()];
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            let mut indices = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let linear = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[linear]);
                indices.push(linear);
            }
            let stats = slice_stats(&slice, &options.method, threshold_factor(options))?;
            let threshold_idx = prefix + suffix * pre;
            lower[threshold_idx] = stats.lower;
            upper[threshold_idx] = stats.upper;
            center[threshold_idx] = stats.center;
            for (value, linear) in slice.iter().zip(indices) {
                mask[linear] = u8::from(is_outlier_value(*value, stats.lower, stats.upper));
                lower_full[linear] = stats.lower;
                upper_full[linear] = stats.upper;
                center_full[linear] = stats.center;
            }
        }
    }
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape.clone(),
        lower,
        upper,
        center,
        lower_full,
        upper_full,
        center_full,
        threshold_shape,
    })
}

fn detect_moving(
    input: &Tensor,
    input_values: &[f64],
    shape: &[usize],
    axis: usize,
    options: &DetectionOptions,
) -> BuiltinResult<DetectionResult> {
    let axis_len = shape[axis];
    let pre: usize = shape[..axis].iter().product();
    let post: usize = shape[axis + 1..].iter().product();
    let mut mask = vec![0u8; input_values.len()];
    let mut lower = vec![f64::NAN; input_values.len()];
    let mut upper = vec![f64::NAN; input_values.len()];
    let mut center = vec![f64::NAN; input_values.len()];
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            let mut indices = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let linear = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[linear]);
                indices.push(linear);
            }
            let window = match options.method {
                DetectionMethod::MovingMedian(window) | DetectionMethod::MovingMean(window) => {
                    window
                }
                _ => unreachable!(),
            };
            for idx in 0..axis_len {
                let (start, end) = centered_window(idx, axis_len, window);
                let stats = slice_stats(
                    &slice[start..end],
                    &options.method,
                    threshold_factor(options),
                )?;
                let linear = indices[idx];
                lower[linear] = stats.lower;
                upper[linear] = stats.upper;
                center[linear] = stats.center;
                mask[linear] = u8::from(is_outlier_value(slice[idx], stats.lower, stats.upper));
            }
        }
    }
    Ok(DetectionResult {
        mask,
        mask_shape: input.shape.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
        center: center.clone(),
        lower_full: lower,
        upper_full: upper,
        center_full: center,
        threshold_shape: shape.to_vec(),
    })
}

fn fill_outlier_tensor(
    input: &Tensor,
    result: &DetectionResult,
    options: &DetectionOptions,
    method: &FillMethod,
) -> BuiltinResult<(Value, Vec<u8>)> {
    let input_values = tensor::tensor_values_f64_cow(input);
    let mut data = input_values.to_vec();
    let mut filled_mask = vec![0; data.len()];
    let shape = tensor::default_shape_for(&input.shape, input_values.len());
    let dim = options.dim.unwrap_or_else(|| first_non_singleton(&shape));
    if dim == 0 {
        fill_slice(
            &mut data,
            &(0..input_values.len()).collect::<Vec<_>>(),
            result,
            method,
            &mut filled_mask,
        );
    } else {
        let axis = dim - 1;
        let rank = shape.len().max(axis + 1);
        let mut padded_shape = shape.clone();
        padded_shape.resize(rank, 1);
        let axis_len = padded_shape[axis];
        let pre: usize = padded_shape[..axis].iter().product();
        let post: usize = padded_shape[axis + 1..].iter().product();
        for prefix in 0..pre {
            for suffix in 0..post {
                let indices = (0..axis_len)
                    .map(|idx| prefix + idx * pre + suffix * pre * axis_len)
                    .collect::<Vec<_>>();
                fill_slice(&mut data, &indices, result, method, &mut filled_mask);
            }
        }
    }
    let output_dtype = match input.numeric_dtype() {
        NumericDType::F32 => NumericDType::F32,
        _ => NumericDType::F64,
    };
    Tensor::new_with_dtype(data, input.shape.clone(), output_dtype)
        .map(tensor::tensor_into_value)
        .map(|value| (value, filled_mask))
        .map_err(|err| internal_error(FILLOUTLIERS_NAME, format!("filloutliers: {err}")))
}

fn fill_slice(
    data: &mut [f64],
    indices: &[usize],
    result: &DetectionResult,
    method: &FillMethod,
    filled_mask: &mut [u8],
) {
    let original = data.to_vec();
    for (pos, linear) in indices.iter().copied().enumerate() {
        if result.mask[linear] == 0 {
            continue;
        }
        let replacement = match method {
            FillMethod::Center => Some(result.center_full[linear]),
            FillMethod::Clip => {
                Some(original[linear].clamp(result.lower_full[linear], result.upper_full[linear]))
            }
            FillMethod::Constant(value) => Some(*value),
            FillMethod::Previous => previous_good(&original, result, indices, pos),
            FillMethod::Next => next_good(&original, result, indices, pos),
            FillMethod::Nearest => Some(
                nearest_good(&original, result, indices, pos).unwrap_or(result.center_full[linear]),
            ),
            FillMethod::Linear => Some(
                linear_interp(&original, result, indices, pos).unwrap_or_else(|| {
                    nearest_good(&original, result, indices, pos)
                        .unwrap_or(result.center_full[linear])
                }),
            ),
        };
        if let Some(replacement) = replacement {
            data[linear] = replacement;
            filled_mask[linear] = 1;
        } else {
            data[linear] = f64::NAN;
        }
    }
}

fn previous_good(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    indices[..pos]
        .iter()
        .rev()
        .copied()
        .find(|idx| result.mask[*idx] == 0 && original[*idx].is_finite())
        .map(|idx| original[idx])
}

fn next_good(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    indices[pos + 1..]
        .iter()
        .copied()
        .find(|idx| result.mask[*idx] == 0 && original[*idx].is_finite())
        .map(|idx| original[idx])
}

fn nearest_good(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    let prev = indices[..pos]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(p, idx)| (pos - p, original[*idx]));
    let next = indices[pos + 1..]
        .iter()
        .enumerate()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(offset, idx)| (offset + 1, original[*idx]));
    match (prev, next) {
        (Some((pd, _)), Some((nd, nv))) if nd < pd => Some(nv),
        (Some((_, pv)), _) => Some(pv),
        (_, Some((_, nv))) => Some(nv),
        _ => None,
    }
}

fn linear_interp(
    original: &[f64],
    result: &DetectionResult,
    indices: &[usize],
    pos: usize,
) -> Option<f64> {
    let prev = indices[..pos]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(p, idx)| (p, original[*idx]));
    let next = indices[pos + 1..]
        .iter()
        .enumerate()
        .find(|(_, idx)| result.mask[**idx] == 0 && original[**idx].is_finite())
        .map(|(offset, idx)| (pos + 1 + offset, original[*idx]));
    match (prev, next) {
        (Some((p, pv)), Some((n, nv))) if n > p => {
            let w = (pos - p) as f64 / (n - p) as f64;
            Some(pv * (1.0 - w) + nv * w)
        }
        (Some((_, pv)), _) => Some(pv),
        (_, Some((_, nv))) => Some(nv),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct SliceStats {
    lower: f64,
    upper: f64,
    center: f64,
}

fn slice_stats(values: &[f64], method: &DetectionMethod, factor: f64) -> BuiltinResult<SliceStats> {
    let mut clean = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if clean.is_empty() {
        return Ok(SliceStats {
            lower: f64::NAN,
            upper: f64::NAN,
            center: f64::NAN,
        });
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    match method {
        DetectionMethod::Median | DetectionMethod::MovingMedian(_) => {
            let center = median_sorted(&clean);
            let mut deviations = clean
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let spread = MAD_SCALE * median_sorted(&deviations);
            Ok(SliceStats {
                lower: center - factor * spread,
                upper: center + factor * spread,
                center,
            })
        }
        DetectionMethod::Mean | DetectionMethod::MovingMean(_) => {
            let center = clean.iter().sum::<f64>() / clean.len() as f64;
            let spread = sample_std(&clean, center);
            Ok(SliceStats {
                lower: center - factor * spread,
                upper: center + factor * spread,
                center,
            })
        }
        DetectionMethod::Quartiles => {
            let q1 = quantile_sorted(&clean, 0.25);
            let q3 = quantile_sorted(&clean, 0.75);
            let iqr = q3 - q1;
            Ok(SliceStats {
                lower: q1 - factor * iqr,
                upper: q3 + factor * iqr,
                center: median_sorted(&clean),
            })
        }
        DetectionMethod::Percentiles(lo, hi) => Ok(SliceStats {
            lower: quantile_sorted(&clean, *lo),
            upper: quantile_sorted(&clean, *hi),
            center: median_sorted(&clean),
        }),
    }
}

fn is_outlier_value(value: f64, lower: f64, upper: f64) -> bool {
    !value.is_nan() && (value < lower || value > upper)
}

fn threshold_factor(options: &DetectionOptions) -> f64 {
    options.threshold_factor.unwrap_or(match options.method {
        DetectionMethod::Quartiles => 1.5,
        DetectionMethod::Percentiles(_, _) => 1.0,
        _ => 3.0,
    })
}

fn centered_window(idx: usize, len: usize, window: usize) -> (usize, usize) {
    let before = window / 2;
    let after = window.saturating_sub(before + 1);
    let start = idx.saturating_sub(before);
    let end = (idx + after + 1).min(len);
    (start, end)
}

fn median_sorted(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn quantile_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let pos = p * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let w = pos - lo as f64;
        values[lo] * (1.0 - w) + values[hi] * w
    }
}

fn sample_std(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}

fn outlier_outputs(
    builtin: &'static str,
    filled: Option<Value>,
    result: DetectionResult,
) -> BuiltinResult<Value> {
    let max_outputs = if filled.is_some() { 5 } else { 4 };
    if matches!(crate::output_count::current_output_count(), Some(n) if n > max_outputs) {
        return Err(too_many_outputs(builtin, max_outputs));
    }
    let mask_shape = result_shape_for_mask(&filled, &result)?;
    let mask_value = Value::LogicalArray(
        LogicalArray::new(result.mask, mask_shape)
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))?,
    );
    let lower = tensor_value(result.lower, result.threshold_shape.clone(), builtin)?;
    let upper = tensor_value(result.upper, result.threshold_shape.clone(), builtin)?;
    let center = tensor_value(result.center, result.threshold_shape, builtin)?;
    let outputs = if let Some(filled) = filled {
        vec![filled, mask_value, lower, upper, center]
    } else {
        vec![mask_value, lower, upper, center]
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        )),
        None => Ok(outputs.into_iter().next().unwrap_or(Value::Num(0.0))),
    }
}

fn result_shape_for_mask(
    filled: &Option<Value>,
    result: &DetectionResult,
) -> BuiltinResult<Vec<usize>> {
    if let Some(Value::Tensor(tensor)) = filled {
        Ok(tensor.shape.clone())
    } else {
        Ok(result.mask_shape.clone())
    }
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>, builtin: &'static str) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn parse_dim(builtin: &'static str, value: &Value) -> BuiltinResult<usize> {
    tensor::parse_dimension(value, builtin).map_err(|err| invalid_argument(builtin, err))
}

fn logical_mask(builtin: &'static str, value: &Value) -> BuiltinResult<(Vec<u8>, Vec<usize>)> {
    match value {
        Value::Bool(flag) => Ok((vec![u8::from(*flag)], vec![1, 1])),
        Value::LogicalArray(mask) => Ok((
            mask.data.iter().map(|flag| u8::from(*flag != 0)).collect(),
            mask.shape.clone(),
        )),
        Value::Tensor(tensor) => Ok((
            tensor
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0 && !value.is_nan()))
                .collect(),
            tensor.shape.clone(),
        )),
        Value::Num(value) => Ok((vec![u8::from(*value != 0.0 && !value.is_nan())], vec![1, 1])),
        Value::Int(value) => Ok((vec![u8::from(!value.is_zero())], vec![1, 1])),
        other => Err(invalid_argument(
            builtin,
            format!("OutlierLocations must be logical or numeric, got {other:?}"),
        )),
    }
}

fn numeric_vector(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(builtin, value.clone())
        .map_err(|err| invalid_argument(builtin, format!("{builtin}: {err}")))?;
    Ok(tensor_values_f64(&tensor))
}

fn scalar_usize(builtin: &'static str, value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(value) = integer_scalar(value) {
        return value.try_to_usize().ok_or_else(|| {
            invalid_argument(builtin, format!("{label} must be a nonnegative integer"))
        });
    }
    let number = scalar_number(value)
        .ok_or_else(|| invalid_argument(builtin, format!("{label} must be numeric")))?;
    if !(number.is_finite() && number >= 0.0 && number.fract() == 0.0) {
        return Err(invalid_argument(
            builtin,
            format!("{label} must be a nonnegative integer"),
        ));
    }
    if number > usize::MAX as f64 || (usize::BITS == 64 && number == usize::MAX as f64) {
        return Err(invalid_argument(
            builtin,
            format!("{label} exceeds platform integer limits"),
        ));
    }
    Ok(number as usize)
}

fn scalar_number(value: &Value) -> Option<f64> {
    if let Some(value) = integer_scalar(value) {
        return Some(value.to_f64());
    }
    match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn first_unrepresentable_usize_double() -> f64 {
        if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        }
    }

    #[test]
    fn isoutlier_documented_floating_gpu_fallback_preserves_explicit_residency() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
                .expect("upload observations");
            runmat_accelerate_api::mark_handle_explicit(&handle);
            let output = block_on(isoutlier_builtin(Value::GpuTensor(handle), Vec::new()))
                .expect("documented resident isoutlier");
            assert!(matches!(output, Value::GpuTensor(_)));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn isoutlier_wgpu_fallback_preserves_explicit_residency() {
        let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("actual WGPU provider");
        let input = Tensor::new(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
            .expect("upload observations");
        runmat_accelerate_api::mark_handle_explicit(&handle);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let output = block_on(isoutlier_builtin(Value::GpuTensor(handle), Vec::new()))
            .expect("documented WGPU isoutlier");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(
            output.device_id,
            runmat_accelerate_api::AccelProvider::device_id(provider)
        );
        assert_eq!(output.shape, vec![5, 1]);
        assert!(runmat_accelerate_api::handle_is_logical(&output));
    }

    #[test]
    fn outlier_numeric_parsers_read_typed_integer_storage_exactly() {
        let wide = u64::MAX - 1;
        assert_eq!(
            numeric_vector(
                ISOUTLIER_NAME,
                &int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), vec![1, 2]),
            )
            .unwrap(),
            vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]
        );
        assert_eq!(
            scalar_number(&int_tensor(IntegerStorage::U64(vec![wide]), vec![1, 1])).unwrap(),
            IntValue::U64(wide).to_f64()
        );
        assert_eq!(
            scalar_usize(
                ISOUTLIER_NAME,
                &int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
                "window",
            )
            .unwrap(),
            4
        );
        assert_eq!(
            logical_mask(
                FILLOUTLIERS_NAME,
                &int_tensor(IntegerStorage::I16(vec![0, -2, 3]), vec![3, 1]),
            )
            .unwrap(),
            (vec![0, 1, 1], vec![3, 1])
        );
    }

    #[test]
    fn outlier_scalar_usize_rejects_negative_typed_integer_storage() {
        let err = scalar_usize(
            ISOUTLIER_NAME,
            &int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
            "window",
        )
        .unwrap_err();
        assert!(
            err.message().contains("nonnegative integer"),
            "{}",
            err.message()
        );
    }

    #[test]
    fn outlier_window_parser_ignores_all_typed_mirrors() {
        let storages = [
            IntegerStorage::I8(vec![4]),
            IntegerStorage::I16(vec![4]),
            IntegerStorage::I32(vec![4]),
            IntegerStorage::I64(vec![4]),
            IntegerStorage::U8(vec![4]),
            IntegerStorage::U16(vec![4]),
            IntegerStorage::U32(vec![4]),
            IntegerStorage::U64(vec![4]),
        ];

        for storage in storages {
            assert_eq!(
                scalar_usize(ISOUTLIER_NAME, &int_tensor(storage, vec![1, 1]), "window",).unwrap(),
                4
            );
        }
    }

    #[test]
    fn outlier_scalar_usize_rejects_unrepresentable_double_boundary() {
        let err = scalar_usize(
            ISOUTLIER_NAME,
            &Value::Num(first_unrepresentable_usize_double()),
            "window",
        )
        .unwrap_err();
        assert!(
            err.message().contains("platform integer limits"),
            "{}",
            err.message()
        );
    }

    #[test]
    fn filloutliers_accepts_typed_integer_input_fill_and_mask() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value =
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
        let fill = poisoned_int_tensor(IntegerStorage::I16(vec![-7]), vec![1, 1], 99.0);
        let locations =
            poisoned_int_tensor(IntegerStorage::U8(vec![0, 0, 1, 0, 0]), vec![5, 1], 0.0);
        let out = block_on(filloutliers_builtin(
            value,
            vec![fill, Value::from("OutlierLocations"), locations],
        ))
        .unwrap();
        assert!(
            matches!(out, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 2.0, -7.0, 4.0, 5.0])
        );
    }

    #[test]
    fn filloutliers_integer_roles_are_independently_gated() {
        let integer_data = int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1]);
        let floating_data = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

        let error = block_on(filloutliers_builtin(
            integer_data,
            vec![Value::from("center")],
        ))
        .expect_err("integer data gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FilloutliersIntegerDataExtension")
        );

        let error = block_on(filloutliers_builtin(
            floating_data.clone(),
            vec![Value::Int(IntValue::I16(-1))],
        ))
        .expect_err("integer fill gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FilloutliersIntegerFillExtension")
        );

        let numeric_mask = tensor(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5, 1]);
        let error = block_on(filloutliers_builtin(
            floating_data,
            vec![
                Value::from("center"),
                Value::from("OutlierLocations"),
                numeric_mask,
            ],
        ))
        .expect_err("numeric mask gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FilloutliersNumericMaskExtension")
        );

        let poison = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let error = block_on(filloutliers_builtin(
            tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]),
            vec![
                Value::from("center"),
                Value::from("ThresholdFactor"),
                poison,
            ],
        ))
        .expect_err("resident argument gate before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FilloutliersResidentInputExtension")
        );
    }

    #[test]
    fn filloutliers_wide_integer_detection_precedes_double_output_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2));
        let base = 9_007_199_254_740_992_u64;
        let value = int_tensor(
            IntegerStorage::U64(vec![base, base + 1, base + 2, base + 3, base + 100]),
            vec![5, 1],
        );
        let output = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
            .expect("wide integer extension");
        let Value::OutputList(values) = output else {
            panic!("expected output list");
        };
        assert!(
            matches!(&values[0], Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 && tensor.integer_storage().is_none())
        );
        assert!(
            matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0, 0, 1])
        );
    }

    #[test]
    fn filloutliers_rejects_integer_span_that_would_collapse_adjacent_wide_deltas() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let exact_limit = 1_u64 << 53;
        let value = int_tensor(
            IntegerStorage::U64(vec![0, exact_limit, exact_limit + 1]),
            vec![3, 1],
        );
        let error = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
            .expect_err("inexact integer span must not be collapsed");
        assert!(error.message().contains("span exceeds 2^53"));
    }

    #[test]
    fn filloutliers_integer_data_extension_covers_all_eight_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![1, 2, 100, 4, 5]),
            IntegerStorage::I16(vec![1, 2, 100, 4, 5]),
            IntegerStorage::I32(vec![1, 2, 100, 4, 5]),
            IntegerStorage::I64(vec![1, 2, 100, 4, 5]),
            IntegerStorage::U8(vec![1, 2, 100, 4, 5]),
            IntegerStorage::U16(vec![1, 2, 100, 4, 5]),
            IntegerStorage::U32(vec![1, 2, 100, 4, 5]),
            IntegerStorage::U64(vec![1, 2, 100, 4, 5]),
        ];
        for storage in storages {
            let value = int_tensor(storage, vec![5, 1]);
            let output = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
                .expect("integer filloutliers class");
            assert!(
                matches!(output, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 && tensor.materialize_f64()[2] == 4.0)
            );
        }
    }

    #[test]
    fn filloutliers_preserves_documented_single_output_class() {
        let value = Value::Tensor(
            Tensor::new_with_dtype(
                vec![1.0, 2.0, 100.0, 4.0, 5.0],
                vec![5, 1],
                NumericDType::F32,
            )
            .unwrap(),
        );
        let output = block_on(filloutliers_builtin(value, vec![Value::from("center")]))
            .expect("single filloutliers");
        assert!(
            matches!(output, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        );
    }

    #[test]
    fn filloutliers_logical_locations_require_exact_shape() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0], vec![2, 2]);
        let locations =
            Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 0], vec![4, 1]).unwrap());
        let error = block_on(filloutliers_builtin(
            value,
            vec![
                Value::from("center"),
                Value::from("OutlierLocations"),
                locations,
            ],
        ))
        .expect_err("mask shape mismatch");
        assert!(error.message().contains("exactly the same shape"));
    }

    #[test]
    fn filloutliers_rejects_logical_locations_with_explicit_find_method() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0], vec![4, 1]);
        let locations =
            Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 0], vec![4, 1]).unwrap());
        let error = block_on(filloutliers_builtin(
            value,
            vec![
                Value::from("center"),
                Value::from("mean"),
                Value::from("OutlierLocations"),
                locations,
            ],
        ))
        .expect_err("find method and known locations are incompatible");
        assert!(error.message().contains("cannot be combined"));
    }

    #[test]
    fn isoutlier_typed_integer_input_reads_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value =
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
        let out = block_on(isoutlier_builtin(value, Vec::new())).unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical mask");
        };
        assert_eq!(mask.data, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn isoutlier_moving_method_reads_typed_integer_input_and_window() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value =
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
        let window = poisoned_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1], 0.0);
        let out = block_on(isoutlier_builtin(
            value,
            vec![Value::from("movmedian"), window],
        ))
        .unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical mask");
        };
        assert_eq!(mask.data, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn isoutlier_threshold_factor_reads_typed_integer_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 9, 10]), vec![5, 1], 0.0);
        let factor = poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1], 0.0);
        let out = block_on(isoutlier_builtin(
            value,
            vec![Value::from("ThresholdFactor"), factor],
        ))
        .unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical mask");
        };
        assert_eq!(mask.data, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn isoutlier_percentile_bounds_read_typed_integer_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value =
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1], 0.0);
        let bounds = poisoned_int_tensor(IntegerStorage::U8(vec![10, 90]), vec![1, 2], 0.0);
        let out = block_on(isoutlier_builtin(
            value,
            vec![Value::from("percentiles"), bounds],
        ))
        .unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical mask");
        };
        assert_eq!(mask.data, vec![1, 0, 1, 0, 0]);
    }

    #[test]
    fn isoutlier_integer_data_is_strictly_gated_and_capabilities_are_declared() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let value = int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]);
        let error = block_on(isoutlier_builtin(value, Vec::new()))
            .expect_err("strict mode rejects typed data");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:IsoutlierIntegerDataExtension")
        );
        assert_eq!(ISOUTLIER_EXTENSIONS.len(), 7);
        assert_eq!(ISOUTLIER_INTEGER_CAPABILITIES.len(), 3);
    }

    #[test]
    fn isoutlier_runmat_integer_data_requires_individually_exact_double_values() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
        let error = block_on(isoutlier_builtin(value, Vec::new()))
            .expect_err("wide integer data cannot cross the floating boundary");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn isoutlier_detects_columnwise_median_outliers() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let out = block_on(isoutlier_builtin(value, Vec::new())).unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical mask");
        };
        assert_eq!(mask.data, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn isoutlier_returns_threshold_outputs() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let out = block_on(isoutlier_builtin(value, vec![Value::from("quartiles")])).unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 4);
        assert!(matches!(&values[0], Value::LogicalArray(mask) if mask.data[2] == 1));
        assert!(matches!(&values[1], Value::Num(_) | Value::Tensor(_)));
    }

    #[test]
    fn filloutliers_supports_center_and_mask_output() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let out = block_on(filloutliers_builtin(value, vec![Value::from("center")])).unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert!(matches!(&values[0], Value::Tensor(tensor) if tensor.materialize_f64()[2] == 4.0));
        assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data[2] == 1));
    }

    #[test]
    fn filloutliers_supports_linear_fill() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let out = block_on(filloutliers_builtin(value, vec![Value::from("linear")])).unwrap();
        assert!(
            matches!(out, Value::Tensor(tensor) if (tensor.materialize_f64()[2] - 3.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn filloutliers_supports_numeric_scalar_constant_fill() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let out = block_on(filloutliers_builtin(value, vec![Value::Num(-1.0)])).unwrap();
        assert!(matches!(out, Value::Tensor(tensor) if tensor.materialize_f64()[2] == -1.0));
    }

    #[test]
    fn filloutliers_uses_per_column_centers() {
        let value = tensor(
            vec![1.0, 2.0, 100.0, 4.0, 10.0, 11.0, 300.0, 13.0],
            vec![4, 2],
        );
        let out = block_on(filloutliers_builtin(value, vec![Value::from("center")])).unwrap();
        assert!(
            matches!(out, Value::Tensor(tensor) if tensor.materialize_f64()[2] == 3.0 && tensor.materialize_f64()[6] == 12.0)
        );
    }

    #[test]
    fn isoutlier_percentiles_work_and_reject_threshold_factor() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let out = block_on(isoutlier_builtin(
            value.clone(),
            vec![
                Value::from("percentiles"),
                tensor(vec![10.0, 90.0], vec![1, 2]),
            ],
        ))
        .unwrap();
        assert!(matches!(out, Value::LogicalArray(mask) if mask.data[0] == 1 && mask.data[2] == 1));
        let err = block_on(isoutlier_builtin(
            value,
            vec![
                Value::from("percentiles"),
                tensor(vec![10.0, 90.0], vec![1, 2]),
                Value::from("ThresholdFactor"),
                Value::Num(2.0),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:outliers:InvalidArgument"));
    }

    #[test]
    fn isoutlier_marks_infinity_when_threshold_is_finite() {
        let value = tensor(vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0], vec![5, 1]);
        let out = block_on(isoutlier_builtin(value, Vec::new())).unwrap();
        assert!(matches!(out, Value::LogicalArray(mask) if mask.data[2] == 1));
    }

    #[test]
    fn moving_even_window_is_backward_weighted_like_matlab() {
        assert_eq!(super::centered_window(3, 8, 4), (1, 5));
    }

    #[test]
    fn previous_and_next_do_not_cross_fill_endpoints() {
        let mask = DetectionResult {
            mask: vec![1, 0, 0],
            mask_shape: vec![3, 1],
            lower: vec![f64::NAN],
            upper: vec![f64::NAN],
            center: vec![f64::NAN],
            lower_full: vec![f64::NAN; 3],
            upper_full: vec![f64::NAN; 3],
            center_full: vec![f64::NAN; 3],
            threshold_shape: vec![1, 1],
        };
        let mut prev = vec![100.0, 2.0, 3.0];
        let mut filled = vec![0; 3];
        fill_slice(
            &mut prev,
            &[0, 1, 2],
            &mask,
            &FillMethod::Previous,
            &mut filled,
        );
        assert!(prev[0].is_nan());
        assert_eq!(filled, vec![0, 0, 0]);
        let mut next = vec![1.0, 2.0, 100.0];
        let mut mask = mask;
        mask.mask = vec![0, 0, 1];
        fill_slice(&mut next, &[0, 1, 2], &mask, &FillMethod::Next, &mut filled);
        assert!(next[2].is_nan());
    }

    #[test]
    fn filloutliers_mask_excludes_detected_endpoint_that_cannot_be_filled() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let value = tensor(vec![100.0, 2.0, 3.0], vec![3, 1]);
        let locations = Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap());
        let output = block_on(filloutliers_builtin(
            value,
            vec![
                Value::from("previous"),
                Value::from("OutlierLocations"),
                locations,
            ],
        ))
        .unwrap();
        let Value::OutputList(values) = output else {
            panic!("expected output list");
        };
        assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0]));
    }

    #[test]
    fn outlier_locations_drive_filloutliers_mask() {
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let locations =
            Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 0, 0], vec![5, 1]).unwrap());
        let out = block_on(filloutliers_builtin(
            value,
            vec![
                Value::from("center"),
                Value::from("OutlierLocations"),
                locations,
            ],
        ))
        .unwrap();
        assert!(matches!(out, Value::Tensor(tensor) if tensor.materialize_f64()[2] == 4.0));
    }

    #[test]
    fn too_many_outputs_error() {
        let _guard = crate::output_count::push_output_count(Some(5));
        let value = tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]);
        let err = block_on(isoutlier_builtin(value, Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:outliers:TooManyOutputs"));
    }
}
