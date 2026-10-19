//! Static argument rules for `fea.PeriodicPairing` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const MATCHING: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MATCHING_PERIODIC_PAIRING_CLASS,
    arguments: &[],
};

pub(super) fn matching(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MATCHING)
}

pub(super) const GEOMETRIC: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_GEOMETRIC_PERIODIC_PAIRING_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new("fea.Length")])],
};

pub(super) fn geometric(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &GEOMETRIC)
}
