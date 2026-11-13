//! Positional contracts for fea.ObservationTarget constructors; shared checks live in
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

const REGION_CLASSES: &[StaticClassIdentity] = &[GEOMETRY_REGION_CLASS];
const POSITION_CLASSES: &[StaticClassIdentity] = &[FEA_POSITION_CLASS];

pub(super) const BOUNDARY: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Object(REGION_CLASSES)], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn boundary(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &BOUNDARY)
}

pub(super) const VOLUME: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Object(REGION_CLASSES)], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn volume(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &VOLUME)
}

pub(super) const POINT: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Object(POSITION_CLASSES)], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn point(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &POINT)
}

pub(super) const PATH: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Positions], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn path(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &PATH)
}

pub(super) const INTERFACE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn interface(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &INTERFACE)
}

pub(super) const PORT: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn port(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &PORT)
}

pub(super) const CONNECTOR: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn connector(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &CONNECTOR)
}

pub(super) const CONSTRAINT: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn constraint(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &CONSTRAINT)
}

pub(super) const WHOLE_MODEL: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_OBSERVATION_TARGET_CLASS);
pub(super) fn whole_model(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &WHOLE_MODEL)
}
