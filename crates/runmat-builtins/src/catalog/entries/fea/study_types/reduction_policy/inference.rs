//! Positional contracts for fea.ReductionPolicy constructors; shared checks live in
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

const SPATIAL_CLASSES: &[StaticClassIdentity] = &[FEA_SPATIAL_REDUCTION_CLASS];
const SERIES_CLASSES: &[StaticClassIdentity] = &[FEA_SERIES_REDUCTION_CLASS];
const TIES_CLASSES: &[StaticClassIdentity] = &[FEA_EXTREMA_TIE_POLICY_CLASS];

pub(super) const NONE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_REDUCTION_POLICY_CLASS);
pub(super) fn none(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NONE)
}

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Object(SPATIAL_CLASSES),
        Object(SERIES_CLASSES),
        Object(TIES_CLASSES),
    ],
    FEA_REDUCTION_POLICY_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
