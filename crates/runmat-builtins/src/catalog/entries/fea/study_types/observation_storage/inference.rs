//! Positional contracts for fea.ObservationStorage constructors; shared checks live in
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

const PRECISION_CLASSES: &[StaticClassIdentity] = &[FEA_SIGNAL_PRECISION_CLASS];
const COMPRESSION_CLASSES: &[StaticClassIdentity] = &[FEA_COMPRESSION_POLICY_CLASS];
const RETENTION_CLASSES: &[StaticClassIdentity] = &[FEA_RETENTION_POLICY_CLASS];
const PARTIAL_CLASSES: &[StaticClassIdentity] = &[FEA_PARTIAL_ACCESS_POLICY_CLASS];

pub(super) const NEW: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Object(PRECISION_CLASSES),
        Object(COMPRESSION_CLASSES),
        Object(RETENTION_CLASSES),
        Object(PARTIAL_CLASSES),
    ],
    FEA_OBSERVATION_STORAGE_CLASS,
);
pub(super) fn new(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NEW)
}
