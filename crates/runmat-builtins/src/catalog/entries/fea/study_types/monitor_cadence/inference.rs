//! Positional contracts for fea.MonitorCadence constructors; shared checks live in
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

pub(super) const ACCEPTED_INCREMENTS: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Count], FEA_MONITOR_CADENCE_CLASS);
pub(super) fn accepted_increments(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &ACCEPTED_INCREMENTS)
}

pub(super) const WALL_SECONDS: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealScalar], FEA_MONITOR_CADENCE_CLASS);
pub(super) fn wall_seconds(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &WALL_SECONDS)
}
