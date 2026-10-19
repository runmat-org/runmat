//! Signature for `fea.SequenceBranch.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "branch",
    BuiltinParamType::Any,
    "Immutable fea.SequenceBranch declaration.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 4] = [
    required("id", BuiltinParamType::StringScalar, "Branch identifier."),
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Producer step identifier.",
    ),
    required(
        "stateIds",
        BuiltinParamType::Any,
        "State id or string/cell array of state ids.",
    ),
    required(
        "consumerStepIds",
        BuiltinParamType::Any,
        "Consumer step id or string/cell array of step ids.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "branch = fea.SequenceBranch.new(id, producerStepId, stateIds, consumerStepIds)",
    &NEW_INPUTS,
    &OUTPUT,
)];
pub const FEA_SEQUENCE_BRANCH_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
