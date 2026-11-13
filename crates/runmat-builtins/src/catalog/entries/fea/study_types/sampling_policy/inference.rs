//! Positional contracts for fea.SamplingPolicy constructors; shared checks live in
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

const EVENT_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_EVENT_CLASS];

pub(super) const FINAL_SAMPLE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn final_sample(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &FINAL_SAMPLE)
}

pub(super) const EVERY_ACCEPTED_INCREMENT: StudyConstructorSignature =
    StudyConstructorSignature::new(&[], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn every_accepted_increment(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &EVERY_ACCEPTED_INCREMENT)
}

pub(super) const TIME_SECONDS: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealValues], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn time_seconds(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &TIME_SECONDS)
}

pub(super) const TIME_INTERVAL_SECONDS: StudyConstructorSignature = StudyConstructorSignature::new(
    &[RealScalar, RealScalar, RealScalar],
    FEA_SAMPLING_POLICY_CLASS,
);
pub(super) fn time_interval_seconds(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &TIME_INTERVAL_SECONDS)
}

pub(super) const FREQUENCY_HERTZ: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealValues], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn frequency_hertz(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &FREQUENCY_HERTZ)
}

pub(super) const FREQUENCY_INTERVAL_HERTZ: StudyConstructorSignature =
    StudyConstructorSignature::new(
        &[RealScalar, RealScalar, RealScalar],
        FEA_SAMPLING_POLICY_CLASS,
    );
pub(super) fn frequency_interval_hertz(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &FREQUENCY_INTERVAL_HERTZ)
}

pub(super) const LOAD_FACTORS: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealValues], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn load_factors(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &LOAD_FACTORS)
}

pub(super) const LOAD_FACTOR_INTERVAL: StudyConstructorSignature = StudyConstructorSignature::new(
    &[RealScalar, RealScalar, RealScalar],
    FEA_SAMPLING_POLICY_CLASS,
);
pub(super) fn load_factor_interval(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &LOAD_FACTOR_INTERVAL)
}

pub(super) const STUDY_PARAMETER_VALUES: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealValues], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn study_parameter_values(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_study_constructor(request, entry, &STUDY_PARAMETER_VALUES)
}

pub(super) const MODE_INDICES: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealValues], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn mode_indices(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &MODE_INDICES)
}

pub(super) const EVENT: StudyConstructorSignature =
    StudyConstructorSignature::new(&[Object(EVENT_CLASSES)], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn event(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &EVENT)
}

pub(super) const ADAPTIVE: StudyConstructorSignature =
    StudyConstructorSignature::new(&[RealScalar, Count, RealScalar], FEA_SAMPLING_POLICY_CLASS);
pub(super) fn adaptive(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &ADAPTIVE)
}
