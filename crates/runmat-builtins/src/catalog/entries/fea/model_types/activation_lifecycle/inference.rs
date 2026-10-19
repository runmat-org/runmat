//! Static argument rules for `fea.ActivationLifecycle` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const ACTIVATE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ActivationLifecycle"),
    arguments: &[],
};

pub(super) fn activate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ACTIVATE)
}

pub(super) const CONTINUE_FROM: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ActivationLifecycle"),
    arguments: &[ModelArg::Text],
};

pub(super) fn continue_from(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CONTINUE_FROM)
}

pub(super) const DEACTIVATE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ActivationLifecycle"),
    arguments: &[ModelArg::Text],
};

pub(super) fn deactivate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &DEACTIVATE)
}

pub(super) const MODIFY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ActivationLifecycle"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.StepObjectRef")]),
    ],
};

pub(super) fn modify(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MODIFY)
}
