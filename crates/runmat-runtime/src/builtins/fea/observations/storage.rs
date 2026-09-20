use super::*;

const REDUCTION_INPUTS: [BuiltinParamDescriptor; 3] = [
    required("spatial", BuiltinParamType::Any, "Typed spatial reduction."),
    required("series", BuiltinParamType::Any, "Typed series reduction."),
    required(
        "extremaTies",
        BuiltinParamType::Any,
        "Typed extrema tie policy.",
    ),
];
descriptor!(
    REDUCTION_DESCRIPTOR,
    REDUCTION_SIGNATURES,
    "policy = fea.ReductionPolicy.new(spatial, series, extremaTies)",
    &REDUCTION_INPUTS
);
descriptor!(
    REDUCTION_NONE_DESCRIPTOR,
    REDUCTION_NONE_SIGNATURES,
    "policy = fea.ReductionPolicy.none()",
    &NO_INPUTS
);
#[runtime_builtin(
    name = "fea.ReductionPolicy.none",
    category = "fea",
    summary = "Preserve the requested observation without reducing it.",
    keywords = "fea,observation,reduction,none,typed",
    descriptor(REDUCTION_NONE_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn reduction_none() -> BuiltinResult<Value> {
    to_typed_value::<Option<ReductionPolicy>>("fea.ReductionPolicy.none", REDUCTION_CLASS, None)
}
#[runtime_builtin(
    name = "fea.ReductionPolicy.new",
    category = "fea",
    summary = "Compose typed spatial and series reductions.",
    keywords = "fea,observation,reduction,typed",
    descriptor(REDUCTION_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn reduction(spatial: Value, series: Value, ties: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ReductionPolicy.new";
    to_typed_value(
        NAME,
        REDUCTION_CLASS,
        Some(ReductionPolicy {
            spatial: policy::spatial_reduction(NAME, &spatial)?,
            series: policy::series_reduction(NAME, &series)?,
            extrema_ties: policy::extrema_ties(NAME, &ties)?,
        }),
    )
}

macro_rules! typed_no_arg {
    ($function:ident, $name:literal, $descriptor:ident, $signatures:ident, $label:literal, $class:expr, $value:expr) => {
        descriptor!($descriptor, $signatures, $label, &NO_INPUTS);
        #[runtime_builtin(name = $name, category = "fea", summary = "Select a typed observation storage policy.", keywords = "fea,observation,storage,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations::storage")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, $class, $value) }
    };
}
typed_no_arg!(
    precision_real32,
    "fea.SignalPrecision.real32",
    PRECISION_REAL32_DESCRIPTOR,
    PRECISION_REAL32_SIGNATURES,
    "value = fea.SignalPrecision.real32()",
    PRECISION_CLASS,
    SignalPrecision::Real32
);
typed_no_arg!(
    precision_real64,
    "fea.SignalPrecision.real64",
    PRECISION_REAL64_DESCRIPTOR,
    PRECISION_REAL64_SIGNATURES,
    "value = fea.SignalPrecision.real64()",
    PRECISION_CLASS,
    SignalPrecision::Real64
);
typed_no_arg!(
    precision_complex64,
    "fea.SignalPrecision.complex64",
    PRECISION_COMPLEX64_DESCRIPTOR,
    PRECISION_COMPLEX64_SIGNATURES,
    "value = fea.SignalPrecision.complex64()",
    PRECISION_CLASS,
    SignalPrecision::Complex64
);
typed_no_arg!(
    precision_complex128,
    "fea.SignalPrecision.complex128",
    PRECISION_COMPLEX128_DESCRIPTOR,
    PRECISION_COMPLEX128_SIGNATURES,
    "value = fea.SignalPrecision.complex128()",
    PRECISION_CLASS,
    SignalPrecision::Complex128
);
typed_no_arg!(
    compression_lossless,
    "fea.CompressionPolicy.lossless",
    COMPRESSION_LOSSLESS_DESCRIPTOR,
    COMPRESSION_LOSSLESS_SIGNATURES,
    "value = fea.CompressionPolicy.lossless()",
    COMPRESSION_CLASS,
    CompressionPolicy::Lossless
);
typed_no_arg!(
    compression_none,
    "fea.CompressionPolicy.none",
    COMPRESSION_NONE_DESCRIPTOR,
    COMPRESSION_NONE_SIGNATURES,
    "value = fea.CompressionPolicy.none()",
    COMPRESSION_CLASS,
    CompressionPolicy::None
);
descriptor!(
    COMPRESSION_ERROR_DESCRIPTOR,
    COMPRESSION_ERROR_SIGNATURES,
    "value = fea.CompressionPolicy.errorBounded(maximumRelativeError)",
    &ONE_VALUE
);
#[runtime_builtin(
    name = "fea.CompressionPolicy.errorBounded",
    category = "fea",
    summary = "Use error-bounded observation compression.",
    keywords = "fea,observation,compression,error,typed",
    descriptor(COMPRESSION_ERROR_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn compression_error(value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.CompressionPolicy.errorBounded";
    to_typed_value(
        NAME,
        COMPRESSION_CLASS,
        CompressionPolicy::ErrorBounded {
            maximum_relative_error: values::finite_numeric_scalar(
                NAME,
                &value,
                "maximumRelativeError",
            )?,
        },
    )
}
typed_no_arg!(
    retention_canonical,
    "fea.RetentionPolicy.canonical",
    RETENTION_CANONICAL_DESCRIPTOR,
    RETENTION_CANONICAL_SIGNATURES,
    "value = fea.RetentionPolicy.canonical()",
    RETENTION_CLASS,
    RetentionPolicy::Canonical
);
typed_no_arg!(
    retention_until,
    "fea.RetentionPolicy.untilDependentsComplete",
    RETENTION_UNTIL_DESCRIPTOR,
    RETENTION_UNTIL_SIGNATURES,
    "value = fea.RetentionPolicy.untilDependentsComplete()",
    RETENTION_CLASS,
    RetentionPolicy::UntilDependentsComplete
);
descriptor!(
    RETENTION_LATEST_DESCRIPTOR,
    RETENTION_LATEST_SIGNATURES,
    "value = fea.RetentionPolicy.latestSamples(count)",
    &LATEST
);
descriptor!(
    RETENTION_EXPLICIT_DESCRIPTOR,
    RETENTION_EXPLICIT_SIGNATURES,
    "value = fea.RetentionPolicy.explicitSamples(maximumCount)",
    &LATEST
);
#[runtime_builtin(
    name = "fea.RetentionPolicy.latestSamples",
    category = "fea",
    summary = "Retain only the latest bounded observation samples.",
    keywords = "fea,observation,retention,bounded,typed",
    descriptor(RETENTION_LATEST_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn retention_latest(count: Value) -> BuiltinResult<Value> {
    to_typed_value(
        "fea.RetentionPolicy.latestSamples",
        RETENTION_CLASS,
        RetentionPolicy::LatestSamples {
            count: usize_from_value("fea.RetentionPolicy.latestSamples", &count)?,
        },
    )
}
#[runtime_builtin(
    name = "fea.RetentionPolicy.explicitSamples",
    category = "fea",
    summary = "Bound explicitly requested retained samples.",
    keywords = "fea,observation,retention,bounded,typed",
    descriptor(RETENTION_EXPLICIT_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn retention_explicit(count: Value) -> BuiltinResult<Value> {
    to_typed_value(
        "fea.RetentionPolicy.explicitSamples",
        RETENTION_CLASS,
        RetentionPolicy::ExplicitSamples {
            maximum_count: usize_from_value("fea.RetentionPolicy.explicitSamples", &count)?,
        },
    )
}

const STORAGE_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "precision",
        BuiltinParamType::Any,
        "Typed signal precision.",
    ),
    required(
        "compression",
        BuiltinParamType::Any,
        "Typed compression policy.",
    ),
    required(
        "retention",
        BuiltinParamType::Any,
        "Typed retention policy.",
    ),
    required(
        "partialAccess",
        BuiltinParamType::Any,
        "Typed partial-access policy.",
    ),
];
descriptor!(
    STORAGE_DESCRIPTOR,
    STORAGE_SIGNATURES,
    "storage = fea.ObservationStorage.new(precision, compression, retention, partialAccess)",
    &STORAGE_INPUTS
);
#[runtime_builtin(
    name = "fea.ObservationStorage.new",
    category = "fea",
    summary = "Compose typed observation storage and retention.",
    keywords = "fea,observation,storage,retention,typed",
    descriptor(STORAGE_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn storage(
    precision: Value,
    compression: Value,
    retention: Value,
    partial: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationStorage.new";
    let precision: TypedStudyValue<SignalPrecision> =
        object_payload(NAME, &precision, PRECISION_CLASS)?;
    let compression: TypedStudyValue<CompressionPolicy> =
        object_payload(NAME, &compression, COMPRESSION_CLASS)?;
    let retention: TypedStudyValue<RetentionPolicy> =
        object_payload(NAME, &retention, RETENTION_CLASS)?;
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        STORAGE_CLASS,
        &ObservationStoragePolicy {
            precision: precision.value,
            compression: compression.value,
            retention: retention.value,
            partial_access: policy::partial_access(NAME, &partial)?,
        },
    )
}

const COST_INPUTS: [BuiltinParamDescriptor; 5] = [
    required(
        "operationCount",
        BuiltinParamType::NumericScalar,
        "Estimated operation count.",
    ),
    required(
        "peakResidentBytes",
        BuiltinParamType::NumericScalar,
        "Estimated peak resident bytes.",
    ),
    required(
        "storedBytes",
        BuiltinParamType::NumericScalar,
        "Estimated stored bytes.",
    ),
    required(
        "transferBytes",
        BuiltinParamType::NumericScalar,
        "Estimated transfer bytes.",
    ),
    required(
        "collectiveCount",
        BuiltinParamType::NumericScalar,
        "Estimated collective count.",
    ),
];
descriptor!(COST_DESCRIPTOR, COST_SIGNATURES, "cost = fea.ObservationCost.new(operationCount, peakResidentBytes, storedBytes, transferBytes, collectiveCount)", &COST_INPUTS);
#[runtime_builtin(
    name = "fea.ObservationCost.new",
    category = "fea",
    summary = "Declare an observation resource-cost estimate.",
    keywords = "fea,observation,cost,resources,typed",
    descriptor(COST_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn cost(args: Vec<Value>) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationCost.new";
    if args.len() != 5 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "expected five cost values",
        ));
    }
    let values = args
        .iter()
        .map(|value| {
            usize_from_value(NAME, value).and_then(|value| {
                u64::try_from(value)
                    .map_err(|_| builtin_error(NAME, &ERROR_INPUT, "cost exceeds u64"))
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    serializable_to_object(
        NAME,
        &ERROR_INTERNAL,
        COST_CLASS,
        &ObservationCostEstimate {
            operation_count: values[0],
            peak_resident_bytes: values[1],
            stored_bytes: values[2],
            transfer_bytes: values[3],
            collective_count: values[4],
        },
    )
}
