//! Signatures for the `fea.ObservationEvent` choices.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "event",
    BuiltinParamType::Any,
    "Immutable fea.ObservationEvent naming a solver, series, or accuracy event.",
)];

const FIRST_YIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "event = fea.ObservationEvent.firstYield()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_EVENT_FIRST_YIELD_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FIRST_YIELD_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const CONTACT_STATUS_CHANGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "event = fea.ObservationEvent.contactStatusChange()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_EVENT_CONTACT_STATUS_CHANGE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&CONTACT_STATUS_CHANGE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const PEAK_RESPONSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "event = fea.ObservationEvent.peakResponse()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_EVENT_PEAK_RESPONSE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&PEAK_RESPONSE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const CONVERGENCE_FAILURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "event = fea.ObservationEvent.convergenceFailure()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_EVENT_CONVERGENCE_FAILURE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&CONVERGENCE_FAILURE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const ACCURACY_CRITERION_CROSSING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "event = fea.ObservationEvent.accuracyCriterionCrossing()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_OBSERVATION_EVENT_ACCURACY_CRITERION_CROSSING_DESCRIPTOR: BuiltinDescriptor =
    descriptor(
        &ACCURACY_CRITERION_CROSSING_SIGNATURES,
        &FEA_STUDY_TYPE_ERRORS,
    );
