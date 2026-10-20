use super::*;
use runmat_analysis_core::ObservationCostPolicy;
use runmat_builtins::fea_contract::study_types::descriptors::{
    compression_policy::*, observation_cost::*, observation_storage::*, reduction_policy::*,
    retention_policy::*, signal_precision::*,
};

#[runtime_builtin(
    name = "fea.ReductionPolicy.none",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn reduction_none() -> BuiltinResult<Value> {
    const NAME: &str = "fea.ReductionPolicy.none";
    study_outputs(&FEA_REDUCTION_POLICY_NONE_DESCRIPTOR, NAME)?;
    absent_typed_value(NAME, REDUCTION_CLASS)
}
#[runtime_builtin(
    name = "fea.ReductionPolicy.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn reduction(spatial: Value, series: Value, ties: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ReductionPolicy.new";
    study_outputs(&FEA_REDUCTION_POLICY_NEW_DESCRIPTOR, NAME)?;
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
    ($function:ident, $name:literal, $descriptor:ident, $class:expr, $value:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::storage")]
        pub async fn $function() -> BuiltinResult<Value> {
            study_outputs(&$descriptor, $name)?;
            to_typed_value($name, $class, $value)
        }
    };
}
typed_no_arg!(
    precision_real32,
    "fea.SignalPrecision.real32",
    FEA_SIGNAL_PRECISION_REAL32_DESCRIPTOR,
    PRECISION_CLASS,
    SignalPrecision::Real32
);
typed_no_arg!(
    precision_real64,
    "fea.SignalPrecision.real64",
    FEA_SIGNAL_PRECISION_REAL64_DESCRIPTOR,
    PRECISION_CLASS,
    SignalPrecision::Real64
);
typed_no_arg!(
    precision_complex64,
    "fea.SignalPrecision.complex64",
    FEA_SIGNAL_PRECISION_COMPLEX64_DESCRIPTOR,
    PRECISION_CLASS,
    SignalPrecision::Complex64
);
typed_no_arg!(
    precision_complex128,
    "fea.SignalPrecision.complex128",
    FEA_SIGNAL_PRECISION_COMPLEX128_DESCRIPTOR,
    PRECISION_CLASS,
    SignalPrecision::Complex128
);
typed_no_arg!(
    compression_lossless,
    "fea.CompressionPolicy.lossless",
    FEA_COMPRESSION_POLICY_LOSSLESS_DESCRIPTOR,
    COMPRESSION_CLASS,
    CompressionPolicy::Lossless
);
typed_no_arg!(
    compression_none,
    "fea.CompressionPolicy.none",
    FEA_COMPRESSION_POLICY_NONE_DESCRIPTOR,
    COMPRESSION_CLASS,
    CompressionPolicy::None
);
#[runtime_builtin(
    name = "fea.CompressionPolicy.errorBounded",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn compression_error(value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.CompressionPolicy.errorBounded";
    study_outputs(&FEA_COMPRESSION_POLICY_ERROR_BOUNDED_DESCRIPTOR, NAME)?;
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
    FEA_RETENTION_POLICY_CANONICAL_DESCRIPTOR,
    RETENTION_CLASS,
    RetentionPolicy::Canonical
);
typed_no_arg!(
    retention_until,
    "fea.RetentionPolicy.untilDependentsComplete",
    FEA_RETENTION_POLICY_UNTIL_DEPENDENTS_COMPLETE_DESCRIPTOR,
    RETENTION_CLASS,
    RetentionPolicy::UntilDependentsComplete
);
#[runtime_builtin(
    name = "fea.RetentionPolicy.latestSamples",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn retention_latest(count: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_RETENTION_POLICY_LATEST_SAMPLES_DESCRIPTOR,
        "fea.RetentionPolicy.latestSamples",
    )?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn retention_explicit(count: Value) -> BuiltinResult<Value> {
    study_outputs(
        &FEA_RETENTION_POLICY_EXPLICIT_SAMPLES_DESCRIPTOR,
        "fea.RetentionPolicy.explicitSamples",
    )?;
    to_typed_value(
        "fea.RetentionPolicy.explicitSamples",
        RETENTION_CLASS,
        RetentionPolicy::ExplicitSamples {
            maximum_count: usize_from_value("fea.RetentionPolicy.explicitSamples", &count)?,
        },
    )
}

#[runtime_builtin(
    name = "fea.ObservationStorage.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn storage(
    precision: Value,
    compression: Value,
    retention: Value,
    partial: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationStorage.new";
    study_outputs(&FEA_OBSERVATION_STORAGE_NEW_DESCRIPTOR, NAME)?;
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

#[runtime_builtin(
    name = "fea.ObservationCost.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::storage"
)]
pub async fn cost(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    const NAME: &str = "fea.ObservationCost.new";
    study_outputs(&FEA_OBSERVATION_COST_NEW_DESCRIPTOR, NAME)?;
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
        &ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: values[0],
            peak_resident_bytes: values[1],
            stored_bytes: values[2],
            transfer_bytes: values[3],
            collective_count: values[4],
        }),
    )
}
