//! Static argument rules for `fea.ElectroThermalStep` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const TRANSIENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectroThermalStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn transient(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSIENT)
}

pub(super) const NONLINEAR: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectroThermalStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn nonlinear(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NONLINEAR)
}
