//! Static argument rules for `fea.ElectroThermalStep` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const TRANSIENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRO_THERMAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn transient(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSIENT)
}

pub(super) const NONLINEAR: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRO_THERMAL_STEP_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn nonlinear(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NONLINEAR)
}
