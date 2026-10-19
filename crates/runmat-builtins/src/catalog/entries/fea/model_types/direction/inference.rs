//! Static argument rules for `fea.Direction` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const VECTOR: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_DIRECTION_CLASS,
    arguments: &[ModelArg::RealVector3],
};

pub(super) fn vector(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &VECTOR)
}
