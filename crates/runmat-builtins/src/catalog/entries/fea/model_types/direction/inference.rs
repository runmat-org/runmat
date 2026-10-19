//! Static argument rules for `fea.Direction` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const VECTOR: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.Direction"),
    arguments: &[ModelArg::RealVector3],
};

pub(super) fn vector(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &VECTOR)
}
