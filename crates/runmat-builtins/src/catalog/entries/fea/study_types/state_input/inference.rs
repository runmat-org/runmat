//! Positional contracts for fea.StateInput constructors; shared checks live in
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
const USE_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_USE_CLASS];
const CONFLICT_CLASSES: &[StaticClassIdentity] = &[FEA_INITIAL_STATE_CONFLICT_POLICY_CLASS];

pub(super) const INITIAL_CONDITIONS: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(KIND_CLASSES),
        Object(USE_CLASSES),
        Object(CONFLICT_CLASSES),
    ],
    FEA_STATE_INPUT_CLASS,
);
pub(super) fn initial_conditions(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &INITIAL_CONDITIONS)
}

pub(super) const UPSTREAM: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(KIND_CLASSES),
        Text,
        Text,
        Object(USE_CLASSES),
        Object(CONFLICT_CLASSES),
    ],
    FEA_STATE_INPUT_CLASS,
);
pub(super) fn upstream(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &UPSTREAM)
}
