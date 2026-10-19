//! Signature for `fea.AnalysisSequence.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "sequence",
    BuiltinParamType::Any,
    "Immutable fea.AnalysisSequence.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 5] = [
    required("id", BuiltinParamType::StringScalar, "Sequence identifier."),
    required(
        "steps",
        BuiltinParamType::Any,
        "Configured procedure step or cell array of steps.",
    ),
    required(
        "dependencies",
        BuiltinParamType::Any,
        "fea.SequenceDependency or cell array (may be empty).",
    ),
    required(
        "transitions",
        BuiltinParamType::Any,
        "fea.StateTransition or cell array (may be empty).",
    ),
    required(
        "branches",
        BuiltinParamType::Any,
        "fea.SequenceBranch or cell array (may be empty).",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "sequence = fea.AnalysisSequence.new(id, steps, dependencies, transitions, branches)",
    &NEW_INPUTS,
    &OUTPUT,
)];
pub const FEA_ANALYSIS_SEQUENCE_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
