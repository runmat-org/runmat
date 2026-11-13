//! Positional contracts for fea.MonitorStopCondition constructors; shared checks live in
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

const QUANTITY_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_QUANTITY_CLASS];

pub(super) const NONE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_MONITOR_STOP_CONDITION_CLASS);
pub(super) fn none(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NONE)
}

pub(super) const LESS_THAN: StudyConstructorSignature = StudyConstructorSignature::new(
    &[Object(QUANTITY_CLASSES), RealScalar],
    FEA_MONITOR_STOP_CONDITION_CLASS,
);
pub(super) fn less_than(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &LESS_THAN)
}

pub(super) const GREATER_THAN: StudyConstructorSignature = StudyConstructorSignature::new(
    &[Object(QUANTITY_CLASSES), RealScalar],
    FEA_MONITOR_STOP_CONDITION_CLASS,
);
pub(super) fn greater_than(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &GREATER_THAN)
}

pub(super) const ABSOLUTE_CHANGE_BELOW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[Object(QUANTITY_CLASSES), RealScalar, Count],
    FEA_MONITOR_STOP_CONDITION_CLASS,
);
pub(super) fn absolute_change_below(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &ABSOLUTE_CHANGE_BELOW)
}
