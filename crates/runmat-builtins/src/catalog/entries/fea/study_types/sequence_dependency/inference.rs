//! Positional contracts for fea.SequenceDependency constructors; shared checks live in
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

const REASON_CLASSES: &[StaticClassIdentity] = &[FEA_SEQUENCE_DEPENDENCY_REASON_CLASS];

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[Text, Text, Object(REASON_CLASSES)],
    FEA_SEQUENCE_DEPENDENCY_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
