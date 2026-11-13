//! Positional contracts for fea.StateTransition constructors; shared checks live in
//! `fea_contract::study_types::inference`.

#[allow(unused_imports)]
use crate::fea_contract::study_types::classes::*;
#[allow(unused_imports)]
use crate::fea_contract::study_types::inference::{
    infer_study_constructor, StudyArgument::*, StudyConstructorSignature, StudyResult,
};
use crate::BuiltinCatalogEntry;
#[allow(unused_imports)]
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

const METHOD_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_TRANSITION_METHOD_CLASS];
const TOPOLOGY_CLASSES: &[StaticClassIdentity] = &[FEA_TOPOLOGY_TRANSITION_POLICY_CLASS];
const REFERENCE_CLASSES: &[StaticClassIdentity] = &[FEA_REFERENCE_STATE_POLICY_CLASS];

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Text,
        Text,
        Text,
        Object(METHOD_CLASSES),
        Object(TOPOLOGY_CLASSES),
        Object(REFERENCE_CLASSES),
    ],
    FEA_STATE_TRANSITION_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
