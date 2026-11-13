//! Signatures for the `fea.RestartPolicy` constructors.

use crate::fea_contract::study_types::{
    descriptor, required, signature, FEA_STUDY_TYPE_ERRORS, NO_INPUTS,
};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "policy",
    BuiltinParamType::Any,
    "Immutable fea.RestartPolicy value.",
)];

const FRESH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.RestartPolicy.fresh()",
    &NO_INPUTS,
    &OUTPUT,
)];
pub const FEA_RESTART_POLICY_FRESH_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&FRESH_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);

const CONTINUE_FROM_COMPLETED_STATE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Completed producer step identifier.",
    ),
    required(
        "stateId",
        BuiltinParamType::StringScalar,
        "Complete producer state identifier.",
    ),
];
const CONTINUE_FROM_COMPLETED_STATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.RestartPolicy.continueFromCompletedState(producerStepId, stateId)",
    &CONTINUE_FROM_COMPLETED_STATE_INPUTS,
    &OUTPUT,
)];
pub const FEA_RESTART_POLICY_CONTINUE_FROM_COMPLETED_STATE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(
        &CONTINUE_FROM_COMPLETED_STATE_SIGNATURES,
        &FEA_STUDY_TYPE_ERRORS,
    );

const RESUME_CHECKPOINT_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "checkpointId",
        BuiltinParamType::StringScalar,
        "Checkpoint identifier.",
    ),
    required(
        "requiredStateDigest",
        BuiltinParamType::StringScalar,
        "Required checkpoint-state digest.",
    ),
];
const RESUME_CHECKPOINT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "policy = fea.RestartPolicy.resumeCheckpoint(checkpointId, requiredStateDigest)",
    &RESUME_CHECKPOINT_INPUTS,
    &OUTPUT,
)];
pub const FEA_RESTART_POLICY_RESUME_CHECKPOINT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&RESUME_CHECKPOINT_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
