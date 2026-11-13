//! Positional contracts for fea.ObservationCost constructors; shared checks live in
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

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[Count, Count, Count, Count, Count],
    FEA_OBSERVATION_COST_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
