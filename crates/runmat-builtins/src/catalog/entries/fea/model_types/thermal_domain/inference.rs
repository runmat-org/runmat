//! Static argument rules for `fea.ThermalDomain` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const AT_REFERENCE_TEMPERATURE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_DOMAIN_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "fea.Temperature",
    )])],
};

pub(super) fn at_reference_temperature(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &AT_REFERENCE_TEMPERATURE)
}
