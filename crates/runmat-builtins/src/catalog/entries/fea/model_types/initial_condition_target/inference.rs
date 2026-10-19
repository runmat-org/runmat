//! Static argument rules for `fea.InitialConditionTarget` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const WHOLE_MODEL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_INITIAL_CONDITION_TARGET_CLASS,
    arguments: &[],
};

pub(super) fn whole_model(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &WHOLE_MODEL)
}
