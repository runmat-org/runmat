//! Positional contracts for fea.CompressionPolicy constructors; shared checks live in
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

pub(super) const LOSSLESS: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_COMPRESSION_POLICY_CLASS);
pub(super) fn lossless(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &LOSSLESS)
}

pub(super) const NONE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_COMPRESSION_POLICY_CLASS);
pub(super) fn none(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &NONE)
}

pub(super) const ERROR_BOUNDED: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealScalar], FEA_COMPRESSION_POLICY_CLASS);
pub(super) fn error_bounded(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &ERROR_BOUNDED)
}
