//! Static argument rules for `fea.ThermalStep` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const STEADY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ThermalStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn steady(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &STEADY)
}

pub(super) const TRANSIENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ThermalStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn transient(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSIENT)
}
