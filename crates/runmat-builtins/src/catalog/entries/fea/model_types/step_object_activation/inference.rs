//! Static argument rules for `fea.StepObjectActivation` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const NEW_ACTIVATION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StepObjectActivation"),
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.StepObjectRef")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ActivationLifecycle")]),
    ],
};

pub(super) fn new_activation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NEW_ACTIVATION)
}
