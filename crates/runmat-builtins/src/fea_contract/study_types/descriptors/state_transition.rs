//! Signature for `fea.StateTransition.new`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "transition",
    BuiltinParamType::Any,
    "Immutable fea.StateTransition declaration.",
)];

const NEW_INPUTS: [BuiltinParamDescriptor; 7] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Transition identifier.",
    ),
    required(
        "producerStepId",
        BuiltinParamType::StringScalar,
        "Producer step identifier.",
    ),
    required(
        "consumerStepId",
        BuiltinParamType::StringScalar,
        "Consumer step identifier.",
    ),
    required(
        "stateId",
        BuiltinParamType::StringScalar,
        "Transferred state identifier.",
    ),
    required(
        "method",
        BuiltinParamType::Any,
        "Typed fea.StateTransitionMethod.",
    ),
    required(
        "topology",
        BuiltinParamType::Any,
        "Typed fea.TopologyTransitionPolicy.",
    ),
    required(
        "reference",
        BuiltinParamType::Any,
        "Typed fea.ReferenceStatePolicy.",
    ),
];
const NEW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("transition = fea.StateTransition.new(id, producerStepId, consumerStepId, stateId, method, topology, reference)", &NEW_INPUTS, &OUTPUT)];
pub const FEA_STATE_TRANSITION_NEW_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&NEW_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
