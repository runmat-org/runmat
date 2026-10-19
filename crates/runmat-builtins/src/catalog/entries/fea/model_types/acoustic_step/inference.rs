//! Static argument rules for `fea.AcousticStep` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const HARMONIC: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.AcousticStep"),
    arguments: &[ModelArg::Text],
};

pub(super) fn harmonic(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &HARMONIC)
}
