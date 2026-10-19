//! Static argument rules for `fea.ElectromagneticDomain` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const AT_FREQUENCY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTROMAGNETIC_DOMAIN_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "fea.Frequency",
    )])],
};

pub(super) fn at_frequency(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &AT_FREQUENCY)
}
