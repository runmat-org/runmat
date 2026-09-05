use std::sync::Once;

use runmat_analysis_core::{
    CompressionPolicy, DiagnosticObservation, ExcitationTarget, MonitorCadence, MonitorObservation,
    MonitorStopCondition, ObservationCostEstimate, ObservationEvent, ObservationId,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationStoragePolicy,
    ObservationTarget, ObservationThreshold, ObservationValueShape, Position, ReductionPolicy,
    RetentionPolicy, SamplingPolicy, SignalPrecision,
};

use super::*;
use crate::builtins::fea::study_policies::observation as policy;
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_class, to_typed_value,
    TypedStudyValue, VALUE_FIELD,
};
use crate::builtins::geometry::region::region_ref_from_value;

pub(super) const TARGET_CLASS: &str = "fea.ObservationTarget";
pub(super) const SAMPLING_CLASS: &str = "fea.SamplingPolicy";
pub(super) const EVENT_CLASS: &str = "fea.ObservationEvent";
pub(super) const REDUCTION_CLASS: &str = "fea.ReductionPolicy";
pub(super) const PRECISION_CLASS: &str = "fea.SignalPrecision";
pub(super) const COMPRESSION_CLASS: &str = "fea.CompressionPolicy";
pub(super) const RETENTION_CLASS: &str = "fea.RetentionPolicy";
pub(super) const STORAGE_CLASS: &str = "fea.ObservationStorage";
pub(super) const COST_CLASS: &str = "fea.ObservationCost";
pub(super) const CADENCE_CLASS: &str = "fea.MonitorCadence";
pub(super) const STOP_CLASS: &str = "fea.MonitorStopCondition";
pub(super) const OBSERVATION_CLASS: &str = "fea.Observation";

const STORAGE_FIELDS: &[&str] = &["precision", "compression", "retention", "partial_access"];
const COST_FIELDS: &[&str] = &[
    "operation_count",
    "peak_resident_bytes",
    "stored_bytes",
    "transfer_bytes",
    "collective_count",
];
const OBSERVATION_FIELDS: &[&str] = &["value"];

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const ONE_VALUE: [BuiltinParamDescriptor; 1] = [required(
    "value",
    BuiltinParamType::NumericScalar,
    "Finite policy value.",
)];
const VALUES: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite sampling coordinates.",
)];
const INTERVAL: [BuiltinParamDescriptor; 3] = [
    required("start", BuiltinParamType::NumericScalar, "Interval start."),
    required("end", BuiltinParamType::NumericScalar, "Interval end."),
    required(
        "interval",
        BuiltinParamType::NumericScalar,
        "Positive interval.",
    ),
];
const ADAPTIVE: [BuiltinParamDescriptor; 3] = [
    required(
        "minimumInterval",
        BuiltinParamType::NumericScalar,
        "Positive minimum interval.",
    ),
    required(
        "maximumSamples",
        BuiltinParamType::NumericScalar,
        "Positive maximum sample count.",
    ),
    required(
        "relativeChange",
        BuiltinParamType::NumericScalar,
        "Positive relative-change threshold.",
    ),
];
const LATEST: [BuiltinParamDescriptor; 1] = [required(
    "count",
    BuiltinParamType::NumericScalar,
    "Positive retained-sample count.",
)];
const NUMERIC_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sampling, storage, or cost value",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Counts and byte estimates remain exact; physical coordinates and tolerances cross once into binary64.",
}];
const NUMERIC_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed observation policy",
        inputs: &NUMERIC_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Observation authoring constructs immutable host values.",
    }];
const STRUCTURAL_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This constructor accepts identifiers and typed FEA values only.",
};

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn numeric_values(name: &'static str, value: &Value) -> BuiltinResult<Vec<f64>> {
    let values = match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        Value::Num(value) => vec![*value],
        Value::Int(value) => vec![value.to_f64()],
        other => {
            return Err(builtin_error(
                name,
                &ERROR_INPUT,
                format!("expected a host numeric scalar or array; got {other:?}"),
            ))
        }
    };
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(builtin_error(
            name,
            &ERROR_INPUT,
            "sampling coordinates must be nonempty and finite",
        ));
    }
    Ok(values)
}

const REGION_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "Typed geometry.Region.",
)];
const ID_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Stable target identifier.",
)];
const POINT_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "position",
    BuiltinParamType::Any,
    "Typed fea.Position.",
)];
const PATH_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "positions",
    BuiltinParamType::NumericArray,
    "N-by-3 path positions in meters.",
)];
descriptor!(
    TARGET_BOUNDARY_DESCRIPTOR,
    TARGET_BOUNDARY_SIGNATURES,
    "target = fea.ObservationTarget.boundary(region)",
    &REGION_TARGET
);
descriptor!(
    TARGET_VOLUME_DESCRIPTOR,
    TARGET_VOLUME_SIGNATURES,
    "target = fea.ObservationTarget.volume(region)",
    &REGION_TARGET
);
descriptor!(
    TARGET_POINT_DESCRIPTOR,
    TARGET_POINT_SIGNATURES,
    "target = fea.ObservationTarget.point(position)",
    &POINT_TARGET
);
descriptor!(
    TARGET_PATH_DESCRIPTOR,
    TARGET_PATH_SIGNATURES,
    "target = fea.ObservationTarget.path(positions)",
    &PATH_TARGET
);
descriptor!(
    TARGET_INTERFACE_DESCRIPTOR,
    TARGET_INTERFACE_SIGNATURES,
    "target = fea.ObservationTarget.interface(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_PORT_DESCRIPTOR,
    TARGET_PORT_SIGNATURES,
    "target = fea.ObservationTarget.port(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_CONNECTOR_DESCRIPTOR,
    TARGET_CONNECTOR_SIGNATURES,
    "target = fea.ObservationTarget.connector(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_CONSTRAINT_DESCRIPTOR,
    TARGET_CONSTRAINT_SIGNATURES,
    "target = fea.ObservationTarget.constraint(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_WHOLE_DESCRIPTOR,
    TARGET_WHOLE_SIGNATURES,
    "target = fea.ObservationTarget.wholeModel()",
    &NO_INPUTS
);

async fn region_target(name: &'static str, region: Value, volume: bool) -> BuiltinResult<Value> {
    let region = region_ref_from_value(name, &region)?;
    let target = if volume {
        RegionTarget::volume(region.region_id)
    } else {
        RegionTarget::boundary(region.region_id)
    };
    to_typed_value(
        name,
        TARGET_CLASS,
        ObservationTarget::Spatial(ExcitationTarget::Region(target)),
    )
}
#[runtime_builtin(
    name = "fea.ObservationTarget.boundary",
    category = "fea",
    summary = "Target a geometry boundary region for observation.",
    keywords = "fea,observation,target,boundary,typed",
    descriptor(TARGET_BOUNDARY_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_boundary(region: Value) -> BuiltinResult<Value> {
    region_target("fea.ObservationTarget.boundary", region, false).await
}
#[runtime_builtin(
    name = "fea.ObservationTarget.volume",
    category = "fea",
    summary = "Target a geometry volume region for observation.",
    keywords = "fea,observation,target,volume,typed",
    descriptor(TARGET_VOLUME_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_volume(region: Value) -> BuiltinResult<Value> {
    region_target("fea.ObservationTarget.volume", region, true).await
}
#[runtime_builtin(
    name = "fea.ObservationTarget.point",
    category = "fea",
    summary = "Target a point for observation.",
    keywords = "fea,observation,target,point,typed",
    descriptor(TARGET_POINT_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_point(position: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationTarget.point";
    let position: Position = object_payload(NAME, &position, values::POSITION_CLASS)?;
    to_typed_value(NAME, TARGET_CLASS, ObservationTarget::Point { position })
}
#[runtime_builtin(
    name = "fea.ObservationTarget.path",
    category = "fea",
    summary = "Target an ordered path for observation.",
    keywords = "fea,observation,target,path,typed",
    descriptor(TARGET_PATH_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_path(positions: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationTarget.path";
    let Value::Tensor(tensor) = positions else {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must be an N-by-3 host numeric matrix",
        ));
    };
    let shape = &tensor.shape;
    if shape.len() != 2 || shape[1] != 3 || shape[0] < 2 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must be an N-by-3 matrix with N at least two",
        ));
    }
    let data = tensor.materialize_f64();
    let mut path = Vec::with_capacity(shape[0]);
    for row in 0..shape[0] {
        path.push(Position::new([
            data[row],
            data[row + shape[0]],
            data[row + 2 * shape[0]],
        ]));
    }
    to_typed_value(
        NAME,
        TARGET_CLASS,
        ObservationTarget::Path { positions: path },
    )
}
macro_rules! id_target {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed named observation target.", keywords = "fea,observation,target,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function(id: String) -> BuiltinResult<Value> {
            to_typed_value($name, TARGET_CLASS, ($value)(nonempty_id($name, "target id", id)?))
        }
    };
}
id_target!(
    target_interface,
    "fea.ObservationTarget.interface",
    TARGET_INTERFACE_DESCRIPTOR,
    |id| ObservationTarget::Spatial(ExcitationTarget::Interface { interface_id: id })
);
id_target!(
    target_port,
    "fea.ObservationTarget.port",
    TARGET_PORT_DESCRIPTOR,
    |id| ObservationTarget::Spatial(ExcitationTarget::Port { port_id: id })
);
id_target!(
    target_connector,
    "fea.ObservationTarget.connector",
    TARGET_CONNECTOR_DESCRIPTOR,
    |id| ObservationTarget::Connector { connector_id: id }
);
id_target!(
    target_constraint,
    "fea.ObservationTarget.constraint",
    TARGET_CONSTRAINT_DESCRIPTOR,
    |id| ObservationTarget::Constraint { constraint_id: id }
);
#[runtime_builtin(
    name = "fea.ObservationTarget.wholeModel",
    category = "fea",
    summary = "Target the whole analysis model for observation.",
    keywords = "fea,observation,target,whole,model,typed",
    descriptor(TARGET_WHOLE_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_whole() -> BuiltinResult<Value> {
    to_typed_value(
        "fea.ObservationTarget.wholeModel",
        TARGET_CLASS,
        ObservationTarget::WholeModel,
    )
}

macro_rules! sampling_descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:expr) => {
        descriptor!($descriptor, $signatures, $label, $inputs);
    };
}
sampling_descriptor!(
    SAMPLING_FINAL_DESCRIPTOR,
    SAMPLING_FINAL_SIGNATURES,
    "policy = fea.SamplingPolicy.finalSample()",
    &NO_INPUTS
);
sampling_descriptor!(
    SAMPLING_INCREMENT_DESCRIPTOR,
    SAMPLING_INCREMENT_SIGNATURES,
    "policy = fea.SamplingPolicy.everyAcceptedIncrement()",
    &NO_INPUTS
);
sampling_descriptor!(
    SAMPLING_TIME_DESCRIPTOR,
    SAMPLING_TIME_SIGNATURES,
    "policy = fea.SamplingPolicy.timeSeconds(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_TIME_INTERVAL_DESCRIPTOR,
    SAMPLING_TIME_INTERVAL_SIGNATURES,
    "policy = fea.SamplingPolicy.timeIntervalSeconds(start, end, interval)",
    &INTERVAL
);
sampling_descriptor!(
    SAMPLING_FREQUENCY_DESCRIPTOR,
    SAMPLING_FREQUENCY_SIGNATURES,
    "policy = fea.SamplingPolicy.frequencyHertz(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_FREQUENCY_INTERVAL_DESCRIPTOR,
    SAMPLING_FREQUENCY_INTERVAL_SIGNATURES,
    "policy = fea.SamplingPolicy.frequencyIntervalHertz(start, end, interval)",
    &INTERVAL
);
sampling_descriptor!(
    SAMPLING_LOAD_DESCRIPTOR,
    SAMPLING_LOAD_SIGNATURES,
    "policy = fea.SamplingPolicy.loadFactors(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_LOAD_INTERVAL_DESCRIPTOR,
    SAMPLING_LOAD_INTERVAL_SIGNATURES,
    "policy = fea.SamplingPolicy.loadFactorInterval(start, end, interval)",
    &INTERVAL
);
sampling_descriptor!(
    SAMPLING_MODES_DESCRIPTOR,
    SAMPLING_MODES_SIGNATURES,
    "policy = fea.SamplingPolicy.modeIndices(values)",
    &VALUES
);
sampling_descriptor!(
    SAMPLING_EVENT_DESCRIPTOR,
    SAMPLING_EVENT_SIGNATURES,
    "policy = fea.SamplingPolicy.event(event)",
    &[required(
        "event",
        BuiltinParamType::Any,
        "Typed observation event."
    )]
);
sampling_descriptor!(
    SAMPLING_ADAPTIVE_DESCRIPTOR,
    SAMPLING_ADAPTIVE_SIGNATURES,
    "policy = fea.SamplingPolicy.adaptive(minimumInterval, maximumSamples, relativeChange)",
    &ADAPTIVE
);

macro_rules! sampling_no_arg {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed observation sampling policy.", keywords = "fea,observation,sampling,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, SAMPLING_CLASS, $value) }
    };
}
sampling_no_arg!(
    sampling_final,
    "fea.SamplingPolicy.finalSample",
    SAMPLING_FINAL_DESCRIPTOR,
    SamplingPolicy::Final
);
sampling_no_arg!(
    sampling_increment,
    "fea.SamplingPolicy.everyAcceptedIncrement",
    SAMPLING_INCREMENT_DESCRIPTOR,
    SamplingPolicy::EveryAcceptedIncrement
);
macro_rules! sampling_values {
    ($function:ident, $name:literal, $descriptor:ident, $wrap:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed observation sampling policy.", keywords = "fea,observation,sampling,typed", descriptor($descriptor), integer_capabilities(NUMERIC_CAPABILITIES), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function(values: Value) -> BuiltinResult<Value> { to_typed_value($name, SAMPLING_CLASS, ($wrap)(numeric_values($name, &values)?)) }
    };
}
sampling_values!(
    sampling_time,
    "fea.SamplingPolicy.timeSeconds",
    SAMPLING_TIME_DESCRIPTOR,
    SamplingPolicy::TimeSeconds
);
sampling_values!(
    sampling_frequency,
    "fea.SamplingPolicy.frequencyHertz",
    SAMPLING_FREQUENCY_DESCRIPTOR,
    SamplingPolicy::FrequencyHertz
);
sampling_values!(
    sampling_load,
    "fea.SamplingPolicy.loadFactors",
    SAMPLING_LOAD_DESCRIPTOR,
    SamplingPolicy::LoadFactors
);
#[runtime_builtin(
    name = "fea.SamplingPolicy.modeIndices",
    category = "fea",
    summary = "Sample selected one-based mode indices.",
    keywords = "fea,observation,sampling,modes,typed",
    descriptor(SAMPLING_MODES_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn sampling_modes(values: Value) -> BuiltinResult<Value> {
    to_typed_value(
        "fea.SamplingPolicy.modeIndices",
        SAMPLING_CLASS,
        SamplingPolicy::ModeIndices(one_based_usize_vec_from_value(
            "fea.SamplingPolicy.modeIndices",
            &values,
        )?),
    )
}
macro_rules! sampling_interval {
    ($function:ident, $name:literal, $descriptor:ident, $wrap:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed interval sampling policy.", keywords = "fea,observation,sampling,interval,typed", descriptor($descriptor), integer_capabilities(NUMERIC_CAPABILITIES), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function(start: Value, end: Value, interval: Value) -> BuiltinResult<Value> {
            to_typed_value(
                $name,
                SAMPLING_CLASS,
                ($wrap)(
                    values::finite_numeric_scalar($name, &start, "start")?,
                    values::finite_numeric_scalar($name, &end, "end")?,
                    values::finite_numeric_scalar($name, &interval, "interval")?,
                ),
            )
        }
    };
}
sampling_interval!(
    sampling_time_interval,
    "fea.SamplingPolicy.timeIntervalSeconds",
    SAMPLING_TIME_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::TimeIntervalSeconds {
        start,
        end,
        interval
    }
);
sampling_interval!(
    sampling_frequency_interval,
    "fea.SamplingPolicy.frequencyIntervalHertz",
    SAMPLING_FREQUENCY_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::FrequencyIntervalHertz {
        start,
        end,
        interval
    }
);
sampling_interval!(
    sampling_load_interval,
    "fea.SamplingPolicy.loadFactorInterval",
    SAMPLING_LOAD_INTERVAL_DESCRIPTOR,
    |start, end, interval| SamplingPolicy::LoadFactorInterval {
        start,
        end,
        interval
    }
);

macro_rules! event_choice {
    ($function:ident, $name:literal, $descriptor:ident, $signatures:ident, $label:literal, $value:expr) => {
        descriptor!($descriptor, $signatures, $label, &NO_INPUTS);
        #[runtime_builtin(name = $name, category = "fea", summary = "Select a typed observation event.", keywords = "fea,observation,event,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function() -> BuiltinResult<Value> { to_typed_value($name, EVENT_CLASS, $value) }
    };
}
event_choice!(
    event_first_yield,
    "fea.ObservationEvent.firstYield",
    EVENT_FIRST_YIELD_DESCRIPTOR,
    EVENT_FIRST_YIELD_SIGNATURES,
    "event = fea.ObservationEvent.firstYield()",
    ObservationEvent::FirstYield
);
event_choice!(
    event_contact,
    "fea.ObservationEvent.contactStatusChange",
    EVENT_CONTACT_DESCRIPTOR,
    EVENT_CONTACT_SIGNATURES,
    "event = fea.ObservationEvent.contactStatusChange()",
    ObservationEvent::ContactStatusChange
);
event_choice!(
    event_peak,
    "fea.ObservationEvent.peakResponse",
    EVENT_PEAK_DESCRIPTOR,
    EVENT_PEAK_SIGNATURES,
    "event = fea.ObservationEvent.peakResponse()",
    ObservationEvent::PeakResponse
);
event_choice!(
    event_failure,
    "fea.ObservationEvent.convergenceFailure",
    EVENT_FAILURE_DESCRIPTOR,
    EVENT_FAILURE_SIGNATURES,
    "event = fea.ObservationEvent.convergenceFailure()",
    ObservationEvent::ConvergenceFailure
);
event_choice!(
    event_accuracy,
    "fea.ObservationEvent.accuracyCriterionCrossing",
    EVENT_ACCURACY_DESCRIPTOR,
    EVENT_ACCURACY_SIGNATURES,
    "event = fea.ObservationEvent.accuracyCriterionCrossing()",
    ObservationEvent::AccuracyCriterionCrossing
);
#[runtime_builtin(
    name = "fea.SamplingPolicy.event",
    category = "fea",
    summary = "Sample when a typed solver event occurs.",
    keywords = "fea,observation,sampling,event,typed",
    descriptor(SAMPLING_EVENT_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn sampling_event(event: Value) -> BuiltinResult<Value> {
    let event: TypedStudyValue<ObservationEvent> =
        object_payload("fea.SamplingPolicy.event", &event, EVENT_CLASS)?;
    to_typed_value(
        "fea.SamplingPolicy.event",
        SAMPLING_CLASS,
        SamplingPolicy::Event(event.value),
    )
}
#[runtime_builtin(
    name = "fea.SamplingPolicy.adaptive",
    category = "fea",
    summary = "Create bounded adaptive observation sampling.",
    keywords = "fea,observation,sampling,adaptive,typed",
    descriptor(SAMPLING_ADAPTIVE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn sampling_adaptive(
    minimum: Value,
    maximum: Value,
    relative: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SamplingPolicy.adaptive";
    to_typed_value(
        NAME,
        SAMPLING_CLASS,
        SamplingPolicy::Adaptive {
            minimum_interval: values::finite_numeric_scalar(NAME, &minimum, "minimumInterval")?,
            maximum_samples: usize_from_value(NAME, &maximum)?,
            relative_change: values::finite_numeric_scalar(NAME, &relative, "relativeChange")?,
        },
    )
}

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
    builtin_path = "crate::builtins::fea::observations"
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
    builtin_path = "crate::builtins::fea::observations"
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
        #[runtime_builtin(name = $name, category = "fea", summary = "Select a typed observation storage policy.", keywords = "fea,observation,storage,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
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
    builtin_path = "crate::builtins::fea::observations"
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
    builtin_path = "crate::builtins::fea::observations"
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
    builtin_path = "crate::builtins::fea::observations"
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
    builtin_path = "crate::builtins::fea::observations"
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
    builtin_path = "crate::builtins::fea::observations"
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

const SPEC_INPUTS: [BuiltinParamDescriptor; 15] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Observation identifier.",
    ),
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed observation quantity.",
    ),
    required("target", BuiltinParamType::Any, "Typed observation target."),
    required(
        "sourceLocation",
        BuiltinParamType::Any,
        "Typed source field location.",
    ),
    required(
        "outputLocation",
        BuiltinParamType::Any,
        "Typed output field location.",
    ),
    required("recovery", BuiltinParamType::Any, "Typed recovery policy."),
    required(
        "component",
        BuiltinParamType::Any,
        "Typed component or invariant.",
    ),
    required("frame", BuiltinParamType::Any, "Typed coordinate frame."),
    required("sign", BuiltinParamType::Any, "Typed sign convention."),
    required(
        "complexRepresentation",
        BuiltinParamType::Any,
        "Typed complex representation.",
    ),
    required("sampling", BuiltinParamType::Any, "Typed sampling policy."),
    required(
        "reduction",
        BuiltinParamType::Any,
        "Typed reduction policy.",
    ),
    required("storage", BuiltinParamType::Any, "Typed storage policy."),
    required(
        "purpose",
        BuiltinParamType::Any,
        "Typed observation purpose.",
    ),
    required(
        "cost",
        BuiltinParamType::Any,
        "Typed observation cost estimate.",
    ),
];
descriptor!(FIELD_DESCRIPTOR, FIELD_SIGNATURES, "observation = fea.Observation.field(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);
descriptor!(HISTORY_DESCRIPTOR, HISTORY_SIGNATURES, "observation = fea.Observation.history(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);
descriptor!(PROBE_DESCRIPTOR, PROBE_SIGNATURES, "observation = fea.Observation.probe(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);
descriptor!(REDUCTION_OBSERVATION_DESCRIPTOR, REDUCTION_OBSERVATION_SIGNATURES, "observation = fea.Observation.reduction(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &SPEC_INPUTS);

fn canonical_shape(quantity: ObservationQuantity) -> ObservationValueShape {
    use ObservationQuantity as Q;
    match quantity {
        Q::Displacement
        | Q::Rotation
        | Q::Velocity
        | Q::Acceleration
        | Q::ReactionForce
        | Q::ReactionMoment
        | Q::TemperatureGradient
        | Q::HeatFlux
        | Q::ElectricField
        | Q::ElectricCurrentDensity
        | Q::MagneticVectorPotential
        | Q::MagneticFluxDensity
        | Q::MagneticFieldIntensity
        | Q::AcousticParticleVelocity
        | Q::FluidVelocity
        | Q::Vorticity
        | Q::WallShearStress => ObservationValueShape::Vector3,
        Q::Strain | Q::Stress => ObservationValueShape::SymmetricTensor3,
        _ => ObservationValueShape::Scalar,
    }
}

enum ObservationForm {
    Field,
    History,
    Probe,
    Reduction,
}
fn observation(
    name: &'static str,
    args: Vec<Value>,
    form: ObservationForm,
) -> BuiltinResult<Value> {
    if args.len() != 15 {
        return Err(builtin_error(
            name,
            &ERROR_INPUT,
            "expected the 15 typed observation arguments",
        ));
    }
    let quantity = policy::quantity(name, &args[1])?;
    let target: TypedStudyValue<ObservationTarget> = object_payload(name, &args[2], TARGET_CLASS)?;
    let sampling: TypedStudyValue<SamplingPolicy> =
        object_payload(name, &args[10], SAMPLING_CLASS)?;
    let spec = ObservationSpec {
        observation_id: ObservationId(nonempty_id(
            name,
            "observation id",
            scalar_string(&args[0], name, &ERROR_INPUT)?,
        )?),
        quantity,
        value_shape: canonical_shape(quantity),
        target: target.value,
        source_location: policy::field_location(name, &args[3])?,
        output_location: policy::field_location(name, &args[4])?,
        recovery: policy::recovery(name, &args[5])?,
        component: policy::component(name, &args[6])?,
        frame: constraints::support::coordinate_frame_from_value(name, &args[7])?,
        sign_convention: policy::sign(name, &args[8])?,
        complex_representation: policy::complex_representation(name, &args[9])?,
        sampling: sampling.value,
        reduction: from_typed_value(name, REDUCTION_CLASS, &args[11])?,
        storage: object_payload(name, &args[12], STORAGE_CLASS)?,
        purpose: policy::purpose(name, &args[13])?,
        cost: object_payload(name, &args[14], COST_CLASS)?,
    };
    let request = match form {
        ObservationForm::Field => ObservationRequest::Field(spec),
        ObservationForm::History => ObservationRequest::History(spec),
        ObservationForm::Probe => ObservationRequest::Probe(spec),
        ObservationForm::Reduction => ObservationRequest::Reduction(spec),
    };
    to_typed_value(name, OBSERVATION_CLASS, request)
}
macro_rules! observation_builtin {
    ($function:ident, $name:literal, $descriptor:ident, $form:expr) => {
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed scientific observation request.", keywords = "fea,observation,field,history,probe,reduction,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function(args: Vec<Value>) -> BuiltinResult<Value> { observation($name, args, $form) }
    };
}
observation_builtin!(
    field,
    "fea.Observation.field",
    FIELD_DESCRIPTOR,
    ObservationForm::Field
);
observation_builtin!(
    history,
    "fea.Observation.history",
    HISTORY_DESCRIPTOR,
    ObservationForm::History
);
observation_builtin!(
    probe,
    "fea.Observation.probe",
    PROBE_DESCRIPTOR,
    ObservationForm::Probe
);
observation_builtin!(
    reduction_observation,
    "fea.Observation.reduction",
    REDUCTION_OBSERVATION_DESCRIPTOR,
    ObservationForm::Reduction
);

const DIAGNOSTIC_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Diagnostic observation identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed diagnostic kind."),
    required("sampling", BuiltinParamType::Any, "Typed sampling policy."),
    required(
        "retention",
        BuiltinParamType::Any,
        "Typed retention policy.",
    ),
];
descriptor!(
    DIAGNOSTIC_DESCRIPTOR,
    DIAGNOSTIC_SIGNATURES,
    "observation = fea.Observation.diagnostic(id, kind, sampling, retention)",
    &DIAGNOSTIC_INPUTS
);
#[runtime_builtin(
    name = "fea.Observation.diagnostic",
    category = "fea",
    summary = "Create a typed diagnostic observation request.",
    keywords = "fea,observation,diagnostic,typed",
    descriptor(DIAGNOSTIC_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn diagnostic(
    id: String,
    kind: Value,
    sampling: Value,
    retention: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.Observation.diagnostic";
    let sampling: TypedStudyValue<SamplingPolicy> =
        object_payload(NAME, &sampling, SAMPLING_CLASS)?;
    let retention: TypedStudyValue<RetentionPolicy> =
        object_payload(NAME, &retention, RETENTION_CLASS)?;
    to_typed_value(
        NAME,
        OBSERVATION_CLASS,
        ObservationRequest::Diagnostic(DiagnosticObservation {
            observation_id: ObservationId(nonempty_id(NAME, "observation id", id)?),
            diagnostic: policy::diagnostic_kind(NAME, &kind)?,
            sampling: sampling.value,
            retention: retention.value,
        }),
    )
}

const CADENCE_ACCEPTED_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interval",
    BuiltinParamType::NumericScalar,
    "Positive accepted-increment interval.",
)];
descriptor!(
    CADENCE_ACCEPTED_DESCRIPTOR,
    CADENCE_ACCEPTED_SIGNATURES,
    "cadence = fea.MonitorCadence.acceptedIncrements(interval)",
    &CADENCE_ACCEPTED_INPUTS
);
descriptor!(
    CADENCE_WALL_DESCRIPTOR,
    CADENCE_WALL_SIGNATURES,
    "cadence = fea.MonitorCadence.wallSeconds(interval)",
    &ONE_VALUE
);
#[runtime_builtin(
    name = "fea.MonitorCadence.acceptedIncrements",
    category = "fea",
    summary = "Evaluate a monitor every bounded number of accepted increments.",
    keywords = "fea,observation,monitor,cadence,typed",
    descriptor(CADENCE_ACCEPTED_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn cadence_accepted(interval: Value) -> BuiltinResult<Value> {
    to_typed_value(
        "fea.MonitorCadence.acceptedIncrements",
        CADENCE_CLASS,
        MonitorCadence::AcceptedIncrements {
            interval: usize_from_value("fea.MonitorCadence.acceptedIncrements", &interval)?,
        },
    )
}
#[runtime_builtin(
    name = "fea.MonitorCadence.wallSeconds",
    category = "fea",
    summary = "Evaluate a monitor at a wall-clock cadence.",
    keywords = "fea,observation,monitor,cadence,typed",
    descriptor(CADENCE_WALL_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn cadence_wall(interval: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorCadence.wallSeconds";
    to_typed_value(
        NAME,
        CADENCE_CLASS,
        MonitorCadence::WallSeconds {
            interval: values::finite_numeric_scalar(NAME, &interval, "interval")?,
        },
    )
}

const STOP_THRESHOLD_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed threshold quantity.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Finite threshold in canonical units.",
    ),
];
const STOP_CHANGE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed threshold quantity.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Finite change threshold in canonical units.",
    ),
    required(
        "consecutive",
        BuiltinParamType::NumericScalar,
        "Positive consecutive-sample count.",
    ),
];
descriptor!(
    STOP_NONE_DESCRIPTOR,
    STOP_NONE_SIGNATURES,
    "condition = fea.MonitorStopCondition.none()",
    &NO_INPUTS
);
descriptor!(
    STOP_LESS_DESCRIPTOR,
    STOP_LESS_SIGNATURES,
    "condition = fea.MonitorStopCondition.lessThan(quantity, canonicalValue)",
    &STOP_THRESHOLD_INPUTS
);
descriptor!(
    STOP_GREATER_DESCRIPTOR,
    STOP_GREATER_SIGNATURES,
    "condition = fea.MonitorStopCondition.greaterThan(quantity, canonicalValue)",
    &STOP_THRESHOLD_INPUTS
);
descriptor!(STOP_CHANGE_DESCRIPTOR, STOP_CHANGE_SIGNATURES, "condition = fea.MonitorStopCondition.absoluteChangeBelow(quantity, canonicalValue, consecutive)", &STOP_CHANGE_INPUTS);
#[runtime_builtin(
    name = "fea.MonitorStopCondition.none",
    category = "fea",
    summary = "Create a progress-only monitor with no stop condition.",
    keywords = "fea,observation,monitor,stop,typed",
    descriptor(STOP_NONE_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_none() -> BuiltinResult<Value> {
    to_typed_value::<Option<MonitorStopCondition>>(
        "fea.MonitorStopCondition.none",
        STOP_CLASS,
        None,
    )
}
fn threshold(
    name: &'static str,
    quantity: &Value,
    value: &Value,
) -> BuiltinResult<ObservationThreshold> {
    Ok(ObservationThreshold {
        quantity: policy::quantity(name, quantity)?,
        canonical_value: values::finite_numeric_scalar(name, value, "canonicalValue")?,
    })
}
#[runtime_builtin(
    name = "fea.MonitorStopCondition.lessThan",
    category = "fea",
    summary = "Stop when a scalar observation is below a typed threshold.",
    keywords = "fea,observation,monitor,stop,threshold,typed",
    descriptor(STOP_LESS_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_less(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.lessThan";
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::LessThan {
            threshold: threshold(NAME, &quantity, &value)?,
        }),
    )
}
#[runtime_builtin(
    name = "fea.MonitorStopCondition.greaterThan",
    category = "fea",
    summary = "Stop when a scalar observation exceeds a typed threshold.",
    keywords = "fea,observation,monitor,stop,threshold,typed",
    descriptor(STOP_GREATER_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_greater(quantity: Value, value: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.greaterThan";
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::GreaterThan {
            threshold: threshold(NAME, &quantity, &value)?,
        }),
    )
}
#[runtime_builtin(
    name = "fea.MonitorStopCondition.absoluteChangeBelow",
    category = "fea",
    summary = "Stop after a scalar observation remains below a change threshold.",
    keywords = "fea,observation,monitor,stop,change,typed",
    descriptor(STOP_CHANGE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn stop_change(
    quantity: Value,
    value: Value,
    consecutive: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.MonitorStopCondition.absoluteChangeBelow";
    to_typed_value(
        NAME,
        STOP_CLASS,
        Some(MonitorStopCondition::AbsoluteChangeBelow {
            threshold: threshold(NAME, &quantity, &value)?,
            consecutive: usize_from_value(NAME, &consecutive)?,
        }),
    )
}

const MONITOR_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Monitor observation identifier.",
    ),
    required(
        "sourceObservationId",
        BuiltinParamType::StringScalar,
        "Scalar source observation identifier.",
    ),
    required("cadence", BuiltinParamType::Any, "Typed monitor cadence."),
    required(
        "stop",
        BuiltinParamType::Any,
        "Typed optional stop condition.",
    ),
];
descriptor!(
    MONITOR_DESCRIPTOR,
    MONITOR_SIGNATURES,
    "observation = fea.Observation.monitor(id, sourceObservationId, cadence, stop)",
    &MONITOR_INPUTS
);
#[runtime_builtin(
    name = "fea.Observation.monitor",
    category = "fea",
    summary = "Create a bounded scalar solver monitor.",
    keywords = "fea,observation,monitor,solver,typed",
    descriptor(MONITOR_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn monitor(
    id: String,
    source: String,
    cadence: Value,
    stop: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.Observation.monitor";
    let cadence: TypedStudyValue<MonitorCadence> = object_payload(NAME, &cadence, CADENCE_CLASS)?;
    let stop: TypedStudyValue<Option<MonitorStopCondition>> =
        object_payload(NAME, &stop, STOP_CLASS)?;
    to_typed_value(
        NAME,
        OBSERVATION_CLASS,
        ObservationRequest::Monitor(MonitorObservation {
            observation_id: ObservationId(nonempty_id(NAME, "monitor id", id)?),
            source_observation_id: ObservationId(nonempty_id(
                NAME,
                "source observation id",
                source,
            )?),
            cadence: cadence.value,
            stop: stop.value,
        }),
    )
}

pub(super) fn observation_requests(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<ObservationRequest>> {
    let wrapped: Vec<TypedStudyValue<ObservationRequest>> =
        object_or_cell(name, value, OBSERVATION_CLASS)?;
    Ok(wrapped.into_iter().map(|value| value.value).collect())
}

pub(super) fn ensure_observation_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            TARGET_CLASS,
            "runmat.fea.observation-target",
            VALUE_FIELD,
            &[
                ("boundary", "fea.ObservationTarget.boundary"),
                ("volume", "fea.ObservationTarget.volume"),
                ("point", "fea.ObservationTarget.point"),
                ("path", "fea.ObservationTarget.path"),
                ("interface", "fea.ObservationTarget.interface"),
                ("port", "fea.ObservationTarget.port"),
                ("connector", "fea.ObservationTarget.connector"),
                ("constraint", "fea.ObservationTarget.constraint"),
                ("wholeModel", "fea.ObservationTarget.wholeModel"),
            ],
        );
        register_immutable_class(
            SAMPLING_CLASS,
            "runmat.fea.sampling-policy",
            VALUE_FIELD,
            &[
                ("finalSample", "fea.SamplingPolicy.finalSample"),
                (
                    "everyAcceptedIncrement",
                    "fea.SamplingPolicy.everyAcceptedIncrement",
                ),
                ("timeSeconds", "fea.SamplingPolicy.timeSeconds"),
                (
                    "timeIntervalSeconds",
                    "fea.SamplingPolicy.timeIntervalSeconds",
                ),
                ("frequencyHertz", "fea.SamplingPolicy.frequencyHertz"),
                (
                    "frequencyIntervalHertz",
                    "fea.SamplingPolicy.frequencyIntervalHertz",
                ),
                ("loadFactors", "fea.SamplingPolicy.loadFactors"),
                (
                    "loadFactorInterval",
                    "fea.SamplingPolicy.loadFactorInterval",
                ),
                ("modeIndices", "fea.SamplingPolicy.modeIndices"),
                ("event", "fea.SamplingPolicy.event"),
                ("adaptive", "fea.SamplingPolicy.adaptive"),
            ],
        );
        register_immutable_class(
            EVENT_CLASS,
            "runmat.fea.observation-event",
            VALUE_FIELD,
            &[
                ("firstYield", "fea.ObservationEvent.firstYield"),
                (
                    "contactStatusChange",
                    "fea.ObservationEvent.contactStatusChange",
                ),
                ("peakResponse", "fea.ObservationEvent.peakResponse"),
                (
                    "convergenceFailure",
                    "fea.ObservationEvent.convergenceFailure",
                ),
                (
                    "accuracyCriterionCrossing",
                    "fea.ObservationEvent.accuracyCriterionCrossing",
                ),
            ],
        );
        register_immutable_class(
            REDUCTION_CLASS,
            "runmat.fea.reduction-policy",
            VALUE_FIELD,
            &[
                ("none", "fea.ReductionPolicy.none"),
                ("new", "fea.ReductionPolicy.new"),
            ],
        );
        register_immutable_class(
            PRECISION_CLASS,
            "runmat.fea.signal-precision",
            VALUE_FIELD,
            &[
                ("real32", "fea.SignalPrecision.real32"),
                ("real64", "fea.SignalPrecision.real64"),
                ("complex64", "fea.SignalPrecision.complex64"),
                ("complex128", "fea.SignalPrecision.complex128"),
            ],
        );
        register_immutable_class(
            COMPRESSION_CLASS,
            "runmat.fea.compression-policy",
            VALUE_FIELD,
            &[
                ("lossless", "fea.CompressionPolicy.lossless"),
                ("errorBounded", "fea.CompressionPolicy.errorBounded"),
                ("none", "fea.CompressionPolicy.none"),
            ],
        );
        register_immutable_class(
            RETENTION_CLASS,
            "runmat.fea.retention-policy",
            VALUE_FIELD,
            &[
                ("canonical", "fea.RetentionPolicy.canonical"),
                (
                    "untilDependentsComplete",
                    "fea.RetentionPolicy.untilDependentsComplete",
                ),
                ("latestSamples", "fea.RetentionPolicy.latestSamples"),
                ("explicitSamples", "fea.RetentionPolicy.explicitSamples"),
            ],
        );
        register_immutable_class(
            STORAGE_CLASS,
            "runmat.fea.observation-storage",
            STORAGE_FIELDS,
            &[("new", "fea.ObservationStorage.new")],
        );
        register_immutable_class(
            COST_CLASS,
            "runmat.fea.observation-cost",
            COST_FIELDS,
            &[("new", "fea.ObservationCost.new")],
        );
        register_immutable_class(
            CADENCE_CLASS,
            "runmat.fea.monitor-cadence",
            VALUE_FIELD,
            &[
                (
                    "acceptedIncrements",
                    "fea.MonitorCadence.acceptedIncrements",
                ),
                ("wallSeconds", "fea.MonitorCadence.wallSeconds"),
            ],
        );
        register_immutable_class(
            STOP_CLASS,
            "runmat.fea.monitor-stop-condition",
            VALUE_FIELD,
            &[
                ("none", "fea.MonitorStopCondition.none"),
                ("lessThan", "fea.MonitorStopCondition.lessThan"),
                ("greaterThan", "fea.MonitorStopCondition.greaterThan"),
                (
                    "absoluteChangeBelow",
                    "fea.MonitorStopCondition.absoluteChangeBelow",
                ),
            ],
        );
        register_immutable_class(
            OBSERVATION_CLASS,
            "runmat.fea.observation",
            OBSERVATION_FIELDS,
            &[
                ("field", "fea.Observation.field"),
                ("history", "fea.Observation.history"),
                ("probe", "fea.Observation.probe"),
                ("reduction", "fea.Observation.reduction"),
                ("diagnostic", "fea.Observation.diagnostic"),
                ("monitor", "fea.Observation.monitor"),
            ],
        );
    });
}
