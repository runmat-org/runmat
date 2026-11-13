//! Static argument rules for `fea.ExcitationCase` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const ALWAYS: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_EXCITATION_CASE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::TypedOrCell(&[
            classes::FEA_STRUCTURAL_EXCITATION_CLASS,
            classes::FEA_THERMAL_EXCITATION_CLASS,
            classes::FEA_ELECTRICAL_EXCITATION_CLASS,
            classes::FEA_MAGNETIC_EXCITATION_CLASS,
            classes::FEA_ACOUSTIC_EXCITATION_CLASS,
            classes::FEA_FLUID_EXCITATION_CLASS,
        ]),
    ],
};

pub(super) fn always(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ALWAYS)
}

pub(super) const DURING_STEPS: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_EXCITATION_CASE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::TypedOrCell(&[
            classes::FEA_STRUCTURAL_EXCITATION_CLASS,
            classes::FEA_THERMAL_EXCITATION_CLASS,
            classes::FEA_ELECTRICAL_EXCITATION_CLASS,
            classes::FEA_MAGNETIC_EXCITATION_CLASS,
            classes::FEA_ACOUSTIC_EXCITATION_CLASS,
            classes::FEA_FLUID_EXCITATION_CLASS,
        ]),
        ModelArg::TextList,
    ],
};

pub(super) fn during_steps(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &DURING_STEPS)
}

pub(super) const MUTUALLY_EXCLUSIVE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_EXCITATION_CASE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::TypedOrCell(&[
            classes::FEA_STRUCTURAL_EXCITATION_CLASS,
            classes::FEA_THERMAL_EXCITATION_CLASS,
            classes::FEA_ELECTRICAL_EXCITATION_CLASS,
            classes::FEA_MAGNETIC_EXCITATION_CLASS,
            classes::FEA_ACOUSTIC_EXCITATION_CLASS,
            classes::FEA_FLUID_EXCITATION_CLASS,
        ]),
        ModelArg::Text,
    ],
};

pub(super) fn mutually_exclusive(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &MUTUALLY_EXCLUSIVE)
}
