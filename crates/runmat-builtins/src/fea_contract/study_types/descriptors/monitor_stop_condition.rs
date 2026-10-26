//! Signatures for the `fea.MonitorStopCondition` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "Immutable fea.MonitorStopCondition value (absent for progress-only monitors).",
)];

const NONE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "condition = fea.MonitorStopCondition.none()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_MONITOR_STOP_CONDITION_NONE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NONE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const LESS_THAN_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity of the monitored source.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Finite threshold in the quantity's canonical SI unit.",
    ),
];
const LESS_THAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "condition = fea.MonitorStopCondition.lessThan(quantity, canonicalValue)",
    &LESS_THAN_INPUTS,
    &OUTPUT,
)];
pub const FEA_MONITOR_STOP_CONDITION_LESS_THAN_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&LESS_THAN_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const GREATER_THAN_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity of the monitored source.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Finite threshold in the quantity's canonical SI unit.",
    ),
];
const GREATER_THAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "condition = fea.MonitorStopCondition.greaterThan(quantity, canonicalValue)",
    &GREATER_THAN_INPUTS,
    &OUTPUT,
)];
pub const FEA_MONITOR_STOP_CONDITION_GREATER_THAN_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&GREATER_THAN_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const ABSOLUTE_CHANGE_BELOW_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "quantity",
        BuiltinParamType::Any,
        "Typed fea.ObservationQuantity of the monitored source.",
    ),
    required(
        "canonicalValue",
        BuiltinParamType::NumericScalar,
        "Nonnegative change threshold in the canonical SI unit.",
    ),
    required(
        "consecutive",
        BuiltinParamType::NumericScalar,
        "Number of consecutive evaluations below the threshold.",
    ),
];
const ABSOLUTE_CHANGE_BELOW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("condition = fea.MonitorStopCondition.absoluteChangeBelow(quantity, canonicalValue, consecutive)", &ABSOLUTE_CHANGE_BELOW_INPUTS, &OUTPUT)];
pub const FEA_MONITOR_STOP_CONDITION_ABSOLUTE_CHANGE_BELOW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&ABSOLUTE_CHANGE_BELOW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
