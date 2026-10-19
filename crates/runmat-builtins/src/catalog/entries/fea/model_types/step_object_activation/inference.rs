//! Static argument rules for `fea.StepObjectActivation` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const NEW_ACTIVATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_ACTIVATION_CLASS,
    arguments: &[
        ModelArg::Typed(&[classes::FEA_STEP_OBJECT_REF_CLASS]),
        ModelArg::Typed(&[classes::FEA_ACTIVATION_LIFECYCLE_CLASS]),
    ],
};

pub(super) fn new_activation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NEW_ACTIVATION)
}
