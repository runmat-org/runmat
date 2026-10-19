//! Static argument rules for `fea.FluidStep` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const STEADY_STATE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn steady_state(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &STEADY_STATE)
}

pub(super) const TRANSIENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn transient(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSIENT)
}
