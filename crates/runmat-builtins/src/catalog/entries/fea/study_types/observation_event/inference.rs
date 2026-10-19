//! Positional contracts for fea.ObservationEvent constructors; shared checks live in
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

pub(super) const FIRST_YIELD: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_OBSERVATION_EVENT_CLASS);
pub(super) fn first_yield(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &FIRST_YIELD)
}

pub(super) const CONTACT_STATUS_CHANGE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_OBSERVATION_EVENT_CLASS);
pub(super) fn contact_status_change(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &CONTACT_STATUS_CHANGE)
}

pub(super) const PEAK_RESPONSE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_OBSERVATION_EVENT_CLASS);
pub(super) fn peak_response(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &PEAK_RESPONSE)
}

pub(super) const CONVERGENCE_FAILURE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_OBSERVATION_EVENT_CLASS);
pub(super) fn convergence_failure(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &CONVERGENCE_FAILURE)
}

pub(super) const ACCURACY_CRITERION_CROSSING: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_OBSERVATION_EVENT_CLASS);
pub(super) fn accuracy_criterion_crossing(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &ACCURACY_CRITERION_CROSSING)
}
