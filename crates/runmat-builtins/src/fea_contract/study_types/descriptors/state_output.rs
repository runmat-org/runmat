//! Signature for `fea.StateOutput.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "output",
    BuiltinParamType::Any,
    "Immutable fea.StateOutput declaration.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Published state identifier.",
    ),
    required("kind", BuiltinParamType::Any, "Typed fea.StateKind."),
    required(
        "completeness",
        BuiltinParamType::Any,
        "Typed fea.StateCompleteness.",
    ),
    required(
        "persistence",
        BuiltinParamType::Any,
        "Typed fea.StatePersistence.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature(
    "output = fea.StateOutput.new(id, kind, completeness, persistence)",
    &NEW_INPUTS,
    &OUTPUT,
)];
pub const FEA_STATE_OUTPUT_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
