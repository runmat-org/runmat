//! Positional contracts for fea.RestartPolicy constructors; shared checks live in
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

pub(super) const FRESH: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_RESTART_POLICY_CLASS);
pub(super) fn fresh(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &FRESH)
}

pub(super) const CONTINUE_FROM_COMPLETED_STATE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text, Text], FEA_RESTART_POLICY_CLASS);
pub(super) fn continue_from_completed_state(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &CONTINUE_FROM_COMPLETED_STATE)
}

pub(super) const RESUME_CHECKPOINT: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text, Text], FEA_RESTART_POLICY_CLASS);
pub(super) fn resume_checkpoint(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &RESUME_CHECKPOINT)
}
