//! Positional contracts for fea.IndependentDomain constructors; shared checks live in
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

const SPECTRA_CLASSES: &[StaticClassIdentity] = &[FEA_RESPONSE_SPECTRUM_CLASS];

pub(super) const STATIC_DOMAIN: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_INDEPENDENT_DOMAIN_CLASS);
pub(super) fn static_domain(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &STATIC_DOMAIN)
}

pub(super) const TIME_SECONDS: StudyConstructorSignature = StudyConstructorSignature::new(
    &[RealScalar, RealScalar, RealScalar, Count],
    FEA_INDEPENDENT_DOMAIN_CLASS,
);
pub(super) fn time_seconds(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &TIME_SECONDS)
}

pub(super) const FREQUENCY_HERTZ: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealValues], FEA_INDEPENDENT_DOMAIN_CLASS);
pub(super) fn frequency_hertz(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &FREQUENCY_HERTZ)
}

pub(super) const LOAD_FACTOR: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealScalar, RealScalar], FEA_INDEPENDENT_DOMAIN_CLASS);
pub(super) fn load_factor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &LOAD_FACTOR)
}

pub(super) const EIGENVALUE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Count], FEA_INDEPENDENT_DOMAIN_CLASS);
pub(super) fn eigenvalue(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &EIGENVALUE)
}

pub(super) const STUDY_PARAMETER: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Text, RealValues], FEA_INDEPENDENT_DOMAIN_CLASS);
pub(super) fn study_parameter(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &STUDY_PARAMETER)
}

pub(super) const RESPONSE_SPECTRA: StudyConstructorSignature =
    StudyConstructorSignature::new(&[ObjectList(SPECTRA_CLASSES)], FEA_INDEPENDENT_DOMAIN_CLASS);
pub(super) fn response_spectra(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &RESPONSE_SPECTRA)
}
