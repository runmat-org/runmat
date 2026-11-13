//! Positional contracts for fea.StateOutput constructors; shared checks live in
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

const KIND_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_KIND_CLASS];
const COMPLETENESS_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_COMPLETENESS_CLASS];
const PERSISTENCE_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_PERSISTENCE_CLASS];

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(KIND_CLASSES),
        Object(COMPLETENESS_CLASSES),
        Object(PERSISTENCE_CLASSES),
    ],
    FEA_STATE_OUTPUT_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
