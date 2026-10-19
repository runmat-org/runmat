//! Static argument rules for `fea.InitialConditionTarget` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const WHOLE_MODEL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.InitialConditionTarget"),
    arguments: &[],
};

pub(super) fn whole_model(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &WHOLE_MODEL)
}
