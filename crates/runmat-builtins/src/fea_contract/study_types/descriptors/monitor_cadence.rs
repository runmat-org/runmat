//! Signatures for the `fea.MonitorCadence` constructors.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "cadence",
    BuiltinParamType::Any,
    "Immutable fea.MonitorCadence value.",
)];

const ACCEPTED_INCREMENTS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interval",
    BuiltinParamType::NumericScalar,
    "Positive number of accepted increments between evaluations.",
)];
const ACCEPTED_INCREMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "cadence = fea.MonitorCadence.acceptedIncrements(interval)",
    &ACCEPTED_INCREMENTS_INPUTS,
    &OUTPUT,
)];
pub const FEA_MONITOR_CADENCE_ACCEPTED_INCREMENTS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&ACCEPTED_INCREMENTS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const WALL_SECONDS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "interval",
    BuiltinParamType::NumericScalar,
    "Positive wall-clock interval in seconds.",
)];
const WALL_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "cadence = fea.MonitorCadence.wallSeconds(interval)",
    &WALL_SECONDS_INPUTS,
    &OUTPUT,
)];
pub const FEA_MONITOR_CADENCE_WALL_SECONDS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&WALL_SECONDS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
