//! Static argument rules for `fea.StructuralStep` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const STATIC_STEP: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn static_step(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &STATIC_STEP)
}

pub(super) const MODAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn modal(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MODAL)
}

pub(super) const HARMONIC: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn harmonic(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &HARMONIC)
}

pub(super) const TRANSIENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn transient(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSIENT)
}

pub(super) const NONLINEAR: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn nonlinear(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NONLINEAR)
}
