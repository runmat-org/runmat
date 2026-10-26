//! Positional contracts for fea.SignalPrecision constructors; shared checks live in
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

pub(super) const REAL32: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_SIGNAL_PRECISION_CLASS);
pub(super) fn real32(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &REAL32)
}

pub(super) const REAL64: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_SIGNAL_PRECISION_CLASS);
pub(super) fn real64(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &REAL64)
}

pub(super) const COMPLEX64: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_SIGNAL_PRECISION_CLASS);
pub(super) fn complex64(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &COMPLEX64)
}

pub(super) const COMPLEX128: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_SIGNAL_PRECISION_CLASS);
pub(super) fn complex128(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &COMPLEX128)
}
