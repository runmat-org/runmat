//! Signatures for the `fea.Observation` request constructors.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "observation",
    BuiltinParamType::Any,
    "Immutable fea.Observation request.",
)];

const FIELD_INPUTS: [BuiltinParamDescriptor; 15] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Observation identifier, unique within its step.",
    ),
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity.",
    ),
    required(
        "target",
        BuiltinParamType::Any,
        "Typed fea.ObservationTarget.",
    ),
    required(
        "sourceLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the quantity is recovered.",
    ),
    required(
        "outputLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the result is reported.",
    ),
    required(
        "recovery",
        BuiltinParamType::Any,
        "Typed fea.RecoveryPolicy.",
    ),
    required(
        "component",
        BuiltinParamType::Any,
        "Typed fea.ObservationComponent or invariant.",
    ),
    required("frame", BuiltinParamType::Any, "Typed fea.CoordinateFrame."),
    required("sign", BuiltinParamType::Any, "Typed fea.SignConvention."),
    required(
        "complexRepresentation",
        BuiltinParamType::Any,
        "Typed fea.ComplexRepresentation.",
    ),
    required(
        "sampling",
        BuiltinParamType::Any,
        "Typed fea.SamplingPolicy.",
    ),
    required(
        "reduction",
        BuiltinParamType::Any,
        "Typed fea.ReductionPolicy.",
    ),
    required(
        "storage",
        BuiltinParamType::Any,
        "Typed fea.ObservationStorage.",
    ),
    required(
        "purpose",
        BuiltinParamType::Any,
        "Typed fea.ObservationPurpose.",
    ),
    required("cost", BuiltinParamType::Any, "Typed fea.ObservationCost."),
];
const FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("observation = fea.Observation.field(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &FIELD_INPUTS, &OUTPUT)];
pub const FEA_OBSERVATION_FIELD_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FIELD_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const HISTORY_INPUTS: [BuiltinParamDescriptor; 15] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Observation identifier, unique within its step.",
    ),
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity.",
    ),
    required(
        "target",
        BuiltinParamType::Any,
        "Typed fea.ObservationTarget.",
    ),
    required(
        "sourceLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the quantity is recovered.",
    ),
    required(
        "outputLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the result is reported.",
    ),
    required(
        "recovery",
        BuiltinParamType::Any,
        "Typed fea.RecoveryPolicy.",
    ),
    required(
        "component",
        BuiltinParamType::Any,
        "Typed fea.ObservationComponent or invariant.",
    ),
    required("frame", BuiltinParamType::Any, "Typed fea.CoordinateFrame."),
    required("sign", BuiltinParamType::Any, "Typed fea.SignConvention."),
    required(
        "complexRepresentation",
        BuiltinParamType::Any,
        "Typed fea.ComplexRepresentation.",
    ),
    required(
        "sampling",
        BuiltinParamType::Any,
        "Typed fea.SamplingPolicy.",
    ),
    required(
        "reduction",
        BuiltinParamType::Any,
        "Typed fea.ReductionPolicy.",
    ),
    required(
        "storage",
        BuiltinParamType::Any,
        "Typed fea.ObservationStorage.",
    ),
    required(
        "purpose",
        BuiltinParamType::Any,
        "Typed fea.ObservationPurpose.",
    ),
    required("cost", BuiltinParamType::Any, "Typed fea.ObservationCost."),
];
const HISTORY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("observation = fea.Observation.history(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &HISTORY_INPUTS, &OUTPUT)];
pub const FEA_OBSERVATION_HISTORY_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&HISTORY_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const PROBE_INPUTS: [BuiltinParamDescriptor; 15] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Observation identifier, unique within its step.",
    ),
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity.",
    ),
    required(
        "target",
        BuiltinParamType::Any,
        "Typed fea.ObservationTarget.",
    ),
    required(
        "sourceLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the quantity is recovered.",
    ),
    required(
        "outputLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the result is reported.",
    ),
    required(
        "recovery",
        BuiltinParamType::Any,
        "Typed fea.RecoveryPolicy.",
    ),
    required(
        "component",
        BuiltinParamType::Any,
        "Typed fea.ObservationComponent or invariant.",
    ),
    required("frame", BuiltinParamType::Any, "Typed fea.CoordinateFrame."),
    required("sign", BuiltinParamType::Any, "Typed fea.SignConvention."),
    required(
        "complexRepresentation",
        BuiltinParamType::Any,
        "Typed fea.ComplexRepresentation.",
    ),
    required(
        "sampling",
        BuiltinParamType::Any,
        "Typed fea.SamplingPolicy.",
    ),
    required(
        "reduction",
        BuiltinParamType::Any,
        "Typed fea.ReductionPolicy.",
    ),
    required(
        "storage",
        BuiltinParamType::Any,
        "Typed fea.ObservationStorage.",
    ),
    required(
        "purpose",
        BuiltinParamType::Any,
        "Typed fea.ObservationPurpose.",
    ),
    required("cost", BuiltinParamType::Any, "Typed fea.ObservationCost."),
];
const PROBE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("observation = fea.Observation.probe(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &PROBE_INPUTS, &OUTPUT)];
pub const FEA_OBSERVATION_PROBE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&PROBE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const REDUCTION_INPUTS: [BuiltinParamDescriptor; 15] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Observation identifier, unique within its step.",
    ),
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity.",
    ),
    required(
        "target",
        BuiltinParamType::Any,
        "Typed fea.ObservationTarget.",
    ),
    required(
        "sourceLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the quantity is recovered.",
    ),
    required(
        "outputLocation",
        BuiltinParamType::Any,
        "Typed fea.FieldLocation where the result is reported.",
    ),
    required(
        "recovery",
        BuiltinParamType::Any,
        "Typed fea.RecoveryPolicy.",
    ),
    required(
        "component",
        BuiltinParamType::Any,
        "Typed fea.ObservationComponent or invariant.",
    ),
    required("frame", BuiltinParamType::Any, "Typed fea.CoordinateFrame."),
    required("sign", BuiltinParamType::Any, "Typed fea.SignConvention."),
    required(
        "complexRepresentation",
        BuiltinParamType::Any,
        "Typed fea.ComplexRepresentation.",
    ),
    required(
        "sampling",
        BuiltinParamType::Any,
        "Typed fea.SamplingPolicy.",
    ),
    required(
        "reduction",
        BuiltinParamType::Any,
        "Typed fea.ReductionPolicy.",
    ),
    required(
        "storage",
        BuiltinParamType::Any,
        "Typed fea.ObservationStorage.",
    ),
    required(
        "purpose",
        BuiltinParamType::Any,
        "Typed fea.ObservationPurpose.",
    ),
    required("cost", BuiltinParamType::Any, "Typed fea.ObservationCost."),
];
const REDUCTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("observation = fea.Observation.reduction(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)", &REDUCTION_INPUTS, &OUTPUT)];
pub const FEA_OBSERVATION_REDUCTION_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&REDUCTION_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const DIAGNOSTIC_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Diagnostic observation identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed fea.DiagnosticKind."),
    required(
        "sampling",
        BuiltinParamType::Any,
        "Typed fea.SamplingPolicy.",
    ),
    required(
        "retention",
        BuiltinParamType::Any,
        "Typed fea.RetentionPolicy.",
    ),
];
const DIAGNOSTIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "observation = fea.Observation.diagnostic(id, kind, sampling, retention)",
    &DIAGNOSTIC_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_DIAGNOSTIC_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&DIAGNOSTIC_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const MONITOR_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Monitor observation identifier.",
    ),
    required(
        "sourceObservationId",
        BuiltinParamType::StringScalar,
        "Id of a scalar observation in the same step.",
    ),
    required(
        "cadence",
        BuiltinParamType::Any,
        "Typed fea.MonitorCadence.",
    ),
    required(
        "stop",
        BuiltinParamType::Any,
        "Typed fea.MonitorStopCondition.",
    ),
];
const MONITOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "observation = fea.Observation.monitor(id, sourceObservationId, cadence, stop)",
    &MONITOR_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_MONITOR_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&MONITOR_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
