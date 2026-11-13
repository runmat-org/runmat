//! Positional contracts for fea.RetentionPolicy constructors; shared checks live in
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

pub(super) const CANONICAL: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_RETENTION_POLICY_CLASS);
pub(super) fn canonical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &CANONICAL)
}

pub(super) const UNTIL_DEPENDENTS_COMPLETE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_RETENTION_POLICY_CLASS);
pub(super) fn until_dependents_complete(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &UNTIL_DEPENDENTS_COMPLETE)
}

pub(super) const LATEST_SAMPLES: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Count], FEA_RETENTION_POLICY_CLASS);
pub(super) fn latest_samples(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &LATEST_SAMPLES)
}

pub(super) const EXPLICIT_SAMPLES: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Count], FEA_RETENTION_POLICY_CLASS);
pub(super) fn explicit_samples(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &EXPLICIT_SAMPLES)
}
