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

mod requests;
mod sampling;
mod storage;
mod targets;

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
