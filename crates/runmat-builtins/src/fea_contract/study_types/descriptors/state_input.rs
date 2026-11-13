//! Signatures for the `fea.StateInput` constructors.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "input",
    BuiltinParamType::Any,
    "Immutable fea.StateInput declaration.",
)];

const INITIAL_CONDITIONS_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "State-input identifier.",
    ),
    required(
        "kind",
        BuiltinParamType::Any,
        "Typed required fea.StateKind.",
    ),
    required("useAs", BuiltinParamType::Any, "Typed fea.StateUse."),
    required(
        "conflict",
        BuiltinParamType::Any,
        "Typed fea.InitialStateConflictPolicy.",
    ),
];
const INITIAL_CONDITIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "input = fea.StateInput.initialConditions(id, kind, useAs, conflict)",
    &INITIAL_CONDITIONS_INPUTS,
    &OUTPUT,
)];
pub const FEA_STATE_INPUT_INITIAL_CONDITIONS_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&INITIAL_CONDITIONS_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const UPSTREAM_INPUTS: [BuiltinParamDescriptor; 6] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "State-input identifier.",
    ),
    required(
        "kind",
        BuiltinParamType::Any,
        "Typed required fea.StateKind.",
    ),
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Producer procedure-step identifier.",
    ),
    required(
        "stateId",
        BuiltinParamType::StringScalar,
        "Producer state identifier.",
    ),
    required("useAs", BuiltinParamType::Any, "Typed fea.StateUse."),
    required(
        "conflict",
        BuiltinParamType::Any,
        "Typed fea.InitialStateConflictPolicy.",
    ),
];
const UPSTREAM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "input = fea.StateInput.upstream(id, kind, producerStepId, stateId, useAs, conflict)",
    &UPSTREAM_INPUTS,
    &OUTPUT,
)];
pub const FEA_STATE_INPUT_UPSTREAM_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&UPSTREAM_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
