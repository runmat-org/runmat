//! Positional contracts for fea.Observation constructors; shared checks live in
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
const TARGET_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_TARGET_CLASS];
const LOCATION_CLASSES: &[StaticClassIdentity] = &[FEA_FIELD_LOCATION_CLASS];
const RECOVERY_CLASSES: &[StaticClassIdentity] = &[FEA_RECOVERY_POLICY_CLASS];
const COMPONENT_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_COMPONENT_CLASS];
const ANY_OBJECT_CLASSES: &[StaticClassIdentity] = &[];
const SIGN_CLASSES: &[StaticClassIdentity] = &[FEA_SIGN_CONVENTION_CLASS];
const COMPLEX_CLASSES: &[StaticClassIdentity] = &[FEA_COMPLEX_REPRESENTATION_CLASS];
const SAMPLING_CLASSES: &[StaticClassIdentity] = &[FEA_SAMPLING_POLICY_CLASS];
const REDUCTION_CLASSES: &[StaticClassIdentity] = &[FEA_REDUCTION_POLICY_CLASS];
const STORAGE_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_STORAGE_CLASS];
const PURPOSE_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_PURPOSE_CLASS];
const COST_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_COST_CLASS];
const DIAGNOSTIC_CLASSES: &[StaticClassIdentity] = &[FEA_DIAGNOSTIC_KIND_CLASS];
const RETENTION_CLASSES: &[StaticClassIdentity] = &[FEA_RETENTION_POLICY_CLASS];
const CADENCE_CLASSES: &[StaticClassIdentity] = &[FEA_MONITOR_CADENCE_CLASS];
const STOP_CLASSES: &[StaticClassIdentity] = &[FEA_MONITOR_STOP_CONDITION_CLASS];

pub(super) const FIELD: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(QUANTITY_CLASSES),
        Object(TARGET_CLASSES),
        Object(LOCATION_CLASSES),
        Object(LOCATION_CLASSES),
        Object(RECOVERY_CLASSES),
        Object(COMPONENT_CLASSES),
        Object(ANY_OBJECT_CLASSES),
        Object(SIGN_CLASSES),
        Object(COMPLEX_CLASSES),
        Object(SAMPLING_CLASSES),
        Object(REDUCTION_CLASSES),
        Object(STORAGE_CLASSES),
        Object(PURPOSE_CLASSES),
        Object(COST_CLASSES),
    ],
    FEA_OBSERVATION_CLASS,
);
pub(super) fn field(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &FIELD)
}

pub(super) const HISTORY: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(QUANTITY_CLASSES),
        Object(TARGET_CLASSES),
        Object(LOCATION_CLASSES),
        Object(LOCATION_CLASSES),
        Object(RECOVERY_CLASSES),
        Object(COMPONENT_CLASSES),
        Object(ANY_OBJECT_CLASSES),
        Object(SIGN_CLASSES),
        Object(COMPLEX_CLASSES),
        Object(SAMPLING_CLASSES),
        Object(REDUCTION_CLASSES),
        Object(STORAGE_CLASSES),
        Object(PURPOSE_CLASSES),
        Object(COST_CLASSES),
    ],
    FEA_OBSERVATION_CLASS,
);
pub(super) fn history(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &HISTORY)
}

pub(super) const PROBE: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(QUANTITY_CLASSES),
        Object(TARGET_CLASSES),
        Object(LOCATION_CLASSES),
        Object(LOCATION_CLASSES),
        Object(RECOVERY_CLASSES),
        Object(COMPONENT_CLASSES),
        Object(ANY_OBJECT_CLASSES),
        Object(SIGN_CLASSES),
        Object(COMPLEX_CLASSES),
        Object(SAMPLING_CLASSES),
        Object(REDUCTION_CLASSES),
        Object(STORAGE_CLASSES),
        Object(PURPOSE_CLASSES),
        Object(COST_CLASSES),
    ],
    FEA_OBSERVATION_CLASS,
);
pub(super) fn probe(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &PROBE)
}

pub(super) const REDUCTION: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(QUANTITY_CLASSES),
        Object(TARGET_CLASSES),
        Object(LOCATION_CLASSES),
        Object(LOCATION_CLASSES),
        Object(RECOVERY_CLASSES),
        Object(COMPONENT_CLASSES),
        Object(ANY_OBJECT_CLASSES),
        Object(SIGN_CLASSES),
        Object(COMPLEX_CLASSES),
        Object(SAMPLING_CLASSES),
        Object(REDUCTION_CLASSES),
        Object(STORAGE_CLASSES),
        Object(PURPOSE_CLASSES),
        Object(COST_CLASSES),
    ],
    FEA_OBSERVATION_CLASS,
);
pub(super) fn reduction(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &REDUCTION)
}

pub(super) const DIAGNOSTIC: StudyConstructorSignature = StudyConstructorSignature::new(
    &[
        Text,
        Object(DIAGNOSTIC_CLASSES),
        Object(SAMPLING_CLASSES),
        Object(RETENTION_CLASSES),
    ],
    FEA_OBSERVATION_CLASS,
);
pub(super) fn diagnostic(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &DIAGNOSTIC)
}

pub(super) const MONITOR: StudyConstructorSignature = StudyConstructorSignature::new(
    &[Text, Text, Object(CADENCE_CLASSES), Object(STOP_CLASSES)],
    FEA_OBSERVATION_CLASS,
);
pub(super) fn monitor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &MONITOR)
}
