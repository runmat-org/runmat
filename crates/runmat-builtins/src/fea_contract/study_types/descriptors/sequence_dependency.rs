//! Signature for `fea.SequenceDependency.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "dependency",
    BuiltinParamType::Any,
    "Immutable fea.SequenceDependency edge.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "predecessorStepId",
        BuiltinParamType::StringScalar,
        "Step that must finish first.",
    ),
    required(
        "successorStepId",
        BuiltinParamType::StringScalar,
        "Step that runs after it.",
    ),
    required(
        "reason",
        BuiltinParamType::Any,
        "Typed fea.SequenceDependencyReason.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "dependency = fea.SequenceDependency.new(predecessorStepId, successorStepId, reason)",
    &NEW_INPUTS,
    &OUTPUT,
)];
pub const FEA_SEQUENCE_DEPENDENCY_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
