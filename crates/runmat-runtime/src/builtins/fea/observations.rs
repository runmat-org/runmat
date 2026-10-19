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
    from_typed_value, nonempty_id, object_or_cell, register_immutable_class, to_typed_value,
    TypedStudyValue, VALUE_FIELD,
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
        object
            .properties
            .insert(VALUE_FIELD[0].to_string(), Value::Tensor(empty));
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
        if object.class_name.display_name() == class_name
            && object
                .properties
                .get(VALUE_FIELD[0])
                .is_none_or(|payload| matches!(payload, Value::Tensor(tensor) if tensor.len() == 0))
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
                (
                    "studyParameterValues",
                    "fea.SamplingPolicy.studyParameterValues",
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
