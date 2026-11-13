use std::sync::Once;

use runmat_analysis_core::{
    AccuracyObservationEvent, CompressionPolicy, DiagnosticObservation, ExcitationTarget,
    MonitorCadence, MonitorObservation, MonitorStopCondition, ObservationCostEstimate,
    ObservationEvent, ObservationId, ObservationQuantity, ObservationRequest, ObservationSpec,
    ObservationStoragePolicy, ObservationTarget, ObservationThreshold, ObservationValueShape,
    Position, ReductionPolicy, RetentionPolicy, SamplingPolicy, SeriesObservationEvent,
    SignalPrecision, SolverObservationEvent,
};

use super::*;
use crate::builtins::fea::study_policies::observation as policy;
use crate::builtins::fea::study_values::{
    from_typed_value, nonempty_id, object_or_cell, register_immutable_runtime_classes,
    register_immutable_schemas, to_typed_value, ImmutableStudyClass, TypedStudyValue, VALUE_FIELD,
};
use crate::builtins::geometry::region::region_ref_from_value;

use runmat_builtins::fea_contract::study_types::{
    classes::{
        FEA_COMPRESSION_POLICY_CLASS, FEA_MONITOR_CADENCE_CLASS, FEA_MONITOR_STOP_CONDITION_CLASS,
        FEA_OBSERVATION_CLASS, FEA_OBSERVATION_COST_CLASS, FEA_OBSERVATION_EVENT_CLASS,
        FEA_OBSERVATION_STORAGE_CLASS, FEA_OBSERVATION_TARGET_CLASS, FEA_REDUCTION_POLICY_CLASS,
        FEA_RETENTION_POLICY_CLASS, FEA_SAMPLING_POLICY_CLASS, FEA_SIGNAL_PRECISION_CLASS,
    },
    FEA_STUDY_TYPE_ERROR_OUTPUTS,
};

pub(super) const TARGET_CLASS: &str = FEA_OBSERVATION_TARGET_CLASS.display_name();
pub(super) const SAMPLING_CLASS: &str = FEA_SAMPLING_POLICY_CLASS.display_name();
pub(super) const EVENT_CLASS: &str = FEA_OBSERVATION_EVENT_CLASS.display_name();
pub(super) const REDUCTION_CLASS: &str = FEA_REDUCTION_POLICY_CLASS.display_name();
pub(super) const PRECISION_CLASS: &str = FEA_SIGNAL_PRECISION_CLASS.display_name();
pub(super) const COMPRESSION_CLASS: &str = FEA_COMPRESSION_POLICY_CLASS.display_name();
pub(super) const RETENTION_CLASS: &str = FEA_RETENTION_POLICY_CLASS.display_name();
pub(super) const STORAGE_CLASS: &str = FEA_OBSERVATION_STORAGE_CLASS.display_name();
pub(super) const COST_CLASS: &str = FEA_OBSERVATION_COST_CLASS.display_name();
pub(super) const CADENCE_CLASS: &str = FEA_MONITOR_CADENCE_CLASS.display_name();
pub(super) const STOP_CLASS: &str = FEA_MONITOR_STOP_CONDITION_CLASS.display_name();
pub(super) const OBSERVATION_CLASS: &str = FEA_OBSERVATION_CLASS.display_name();

const STORAGE_FIELDS: &[&str] = &["precision", "compression", "retention", "partial_access"];
/// `fea.ObservationCost` is the tagged `ObservationCostPolicy` (`mode` + `limit`).
const COST_FIELDS: &[&str] = &["mode", "limit"];
const OBSERVATION_FIELDS: &[&str] = &["value"];

/// Typed study constructors return exactly one value; the catalog descriptor
/// is the single authority for that bound.
pub(in crate::builtins::fea) fn study_outputs(
    descriptor: &'static BuiltinDescriptor,
    builtin: &'static str,
) -> BuiltinResult<()> {
    crate::output_count::validate_fixed_outputs(descriptor, &FEA_STUDY_TYPE_ERROR_OUTPUTS, builtin)
}

/// Absent typed choice such as `fea.ReductionPolicy.none()`: the registered
/// `value` property holds an empty double, so `isempty(policy.value)` is true
/// and the object satisfies its immutable-value schema.
fn absent_typed_value(builtin: &'static str, class_name: &'static str) -> BuiltinResult<Value> {
    let mut value = to_typed_value::<Option<()>>(builtin, class_name, None)?;
    if let Value::Object(object) = &mut value {
        let empty = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|message| builtin_error(builtin, &ERROR_INTERNAL, message))?;
        crate::builtins::fea::object_api::replace(
            builtin,
            object,
            VALUE_FIELD[0],
            Value::Tensor(empty),
        )?;
    }
    Ok(value)
}

/// Decode an optional typed choice; an empty (or missing) `value` is absent.
fn optional_typed_value<T: DeserializeOwned>(
    builtin: &'static str,
    class_name: &'static str,
    value: &Value,
) -> BuiltinResult<Option<T>> {
    if let Value::Object(object) = value {
        if object.class_name().display_name() == class_name
            && object
                .properties()
                .get(VALUE_FIELD[0])
                .is_none_or(|payload| matches!(payload, Value::Tensor(tensor) if tensor.is_empty()))
        {
            return Ok(None);
        }
    }
    from_typed_value::<Option<T>>(builtin, class_name, value)
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

pub(crate) mod requests;
pub(crate) mod sampling;
pub(crate) mod storage;
pub(crate) mod targets;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(super) use requests::field;
pub(crate) use requests::observation_requests;
#[cfg(test)]
pub(super) use sampling::sampling_final;
#[cfg(test)]
pub(super) use storage::{
    compression_none, cost, precision_real64, reduction_none, retention_canonical, storage,
};
#[cfg(test)]
pub(super) use targets::target_whole;

const OBSERVATION_STUDY_CLASSES: [ImmutableStudyClass; 12] = [
    ImmutableStudyClass {
        class_name: TARGET_CLASS,
        type_identity: "runmat.fea.observation-target",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: SAMPLING_CLASS,
        type_identity: "runmat.fea.sampling-policy",
        fields: VALUE_FIELD,
        methods: &[
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
            (
                "studyParameterValues",
                "fea.SamplingPolicy.studyParameterValues",
            ),
            ("modeIndices", "fea.SamplingPolicy.modeIndices"),
            ("event", "fea.SamplingPolicy.event"),
            ("adaptive", "fea.SamplingPolicy.adaptive"),
        ],
    },
    ImmutableStudyClass {
        class_name: EVENT_CLASS,
        type_identity: "runmat.fea.observation-event",
        fields: VALUE_FIELD,
        methods: &[
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
    },
    ImmutableStudyClass {
        class_name: REDUCTION_CLASS,
        type_identity: "runmat.fea.reduction-policy",
        fields: VALUE_FIELD,
        methods: &[
            ("none", "fea.ReductionPolicy.none"),
            ("new", "fea.ReductionPolicy.new"),
        ],
    },
    ImmutableStudyClass {
        class_name: PRECISION_CLASS,
        type_identity: "runmat.fea.signal-precision",
        fields: VALUE_FIELD,
        methods: &[
            ("real32", "fea.SignalPrecision.real32"),
            ("real64", "fea.SignalPrecision.real64"),
            ("complex64", "fea.SignalPrecision.complex64"),
            ("complex128", "fea.SignalPrecision.complex128"),
        ],
    },
    ImmutableStudyClass {
        class_name: COMPRESSION_CLASS,
        type_identity: "runmat.fea.compression-policy",
        fields: VALUE_FIELD,
        methods: &[
            ("lossless", "fea.CompressionPolicy.lossless"),
            ("errorBounded", "fea.CompressionPolicy.errorBounded"),
            ("none", "fea.CompressionPolicy.none"),
        ],
    },
    ImmutableStudyClass {
        class_name: RETENTION_CLASS,
        type_identity: "runmat.fea.retention-policy",
        fields: VALUE_FIELD,
        methods: &[
            ("canonical", "fea.RetentionPolicy.canonical"),
            (
                "untilDependentsComplete",
                "fea.RetentionPolicy.untilDependentsComplete",
            ),
            ("latestSamples", "fea.RetentionPolicy.latestSamples"),
            ("explicitSamples", "fea.RetentionPolicy.explicitSamples"),
        ],
    },
    ImmutableStudyClass {
        class_name: STORAGE_CLASS,
        type_identity: "runmat.fea.observation-storage",
        fields: STORAGE_FIELDS,
        methods: &[("new", "fea.ObservationStorage.new")],
    },
    ImmutableStudyClass {
        class_name: COST_CLASS,
        type_identity: "runmat.fea.observation-cost",
        fields: COST_FIELDS,
        methods: &[("new", "fea.ObservationCost.new")],
    },
    ImmutableStudyClass {
        class_name: CADENCE_CLASS,
        type_identity: "runmat.fea.monitor-cadence",
        fields: VALUE_FIELD,
        methods: &[
            (
                "acceptedIncrements",
                "fea.MonitorCadence.acceptedIncrements",
            ),
            ("wallSeconds", "fea.MonitorCadence.wallSeconds"),
        ],
    },
    ImmutableStudyClass {
        class_name: STOP_CLASS,
        type_identity: "runmat.fea.monitor-stop-condition",
        fields: VALUE_FIELD,
        methods: &[
            ("none", "fea.MonitorStopCondition.none"),
            ("lessThan", "fea.MonitorStopCondition.lessThan"),
            ("greaterThan", "fea.MonitorStopCondition.greaterThan"),
            (
                "absoluteChangeBelow",
                "fea.MonitorStopCondition.absoluteChangeBelow",
            ),
        ],
    },
    ImmutableStudyClass {
        class_name: OBSERVATION_CLASS,
        type_identity: "runmat.fea.observation",
        fields: OBSERVATION_FIELDS,
        methods: &[
            ("field", "fea.Observation.field"),
            ("history", "fea.Observation.history"),
            ("probe", "fea.Observation.probe"),
            ("reduction", "fea.Observation.reduction"),
            ("diagnostic", "fea.Observation.diagnostic"),
            ("monitor", "fea.Observation.monitor"),
        ],
    },
];

pub(super) fn ensure_observation_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(FEA_OBSERVATION_TARGET_CLASS);
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &OBSERVATION_STUDY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_immutable_runtime_classes(&OBSERVATION_STUDY_CLASSES);
    });
}
