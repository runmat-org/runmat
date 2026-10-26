//! Positional contracts for fea.ProcedureStep constructors; shared checks live in
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

const ACTIVATION_CLASSES: &[StaticClassIdentity] = &[FEA_STEP_OBJECT_ACTIVATION_CLASS];
const STATE_INPUT_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_INPUT_CLASS];
const STATE_OUTPUT_CLASSES: &[StaticClassIdentity] = &[FEA_STATE_OUTPUT_CLASS];
const OBSERVATION_CLASSES: &[StaticClassIdentity] = &[FEA_OBSERVATION_CLASS];
const DOMAIN_CLASSES: &[StaticClassIdentity] = &[FEA_INDEPENDENT_DOMAIN_CLASS];
const ACCURACY_CLASSES: &[StaticClassIdentity] = &[FEA_ACCURACY_PLAN_CLASS];
const RESTART_CLASSES: &[StaticClassIdentity] = &[FEA_RESTART_POLICY_CLASS];

pub(super) const CONFIGURE: StudyConstructorSignature = StudyConstructorSignature {
    arguments: &[
        Object(&FEA_PROCEDURE_STEP_CLASSES),
        Object(DOMAIN_CLASSES),
        ObjectList(ACTIVATION_CLASSES),
        ObjectList(STATE_INPUT_CLASSES),
        ObjectList(STATE_OUTPUT_CLASSES),
        ObjectList(OBSERVATION_CLASSES),
        Object(ACCURACY_CLASSES),
        Object(RESTART_CLASSES),
    ],
    result: StudyResult::SameClassAs {
        argument: 0,
        admitted: &FEA_PROCEDURE_STEP_CLASSES,
    },
};
pub(super) fn configure(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_study_constructor(request, entry, &CONFIGURE)
}
