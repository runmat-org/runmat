//! Positional contracts for fea.AnalysisSequence constructors; shared checks live in
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

const DEPENDENCY_CLASSES: &[StaticClassIdentity] = &[FEA_SEQUENCE_DEPENDENCY_CLASS];
const TRANSITION_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_TRANSITION_CLASS];
const BRANCH_CLASSES: &[StaticClassIdentity] = &[FEA_SEQUENCE_BRANCH_CLASS];

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        ObjectList(&FEA_PROCEDURE_STEP_CLASSES),
        ObjectList(DEPENDENCY_CLASSES),
        ObjectList(TRANSITION_CLASSES),
        ObjectList(BRANCH_CLASSES),
    ],
    FEA_ANALYSIS_SEQUENCE_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
