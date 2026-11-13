//! Static argument rules for `fea.ActivationLifecycle` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const ACTIVATE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACTIVATION_LIFECYCLE_CLASS,
    arguments: &[],
};

pub(super) fn activate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ACTIVATE)
}

pub(super) const CONTINUE_FROM: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACTIVATION_LIFECYCLE_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn continue_from(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CONTINUE_FROM)
}

pub(super) const DEACTIVATE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACTIVATION_LIFECYCLE_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn deactivate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &DEACTIVATE)
}

pub(super) const MODIFY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACTIVATION_LIFECYCLE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[classes::FEA_STEP_OBJECT_REF_CLASS]),
    ],
};

pub(super) fn modify(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MODIFY)
}
