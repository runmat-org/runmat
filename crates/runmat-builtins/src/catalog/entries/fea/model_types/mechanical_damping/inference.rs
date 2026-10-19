//! Static argument rules for `fea.MechanicalDamping` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const UNDAMPED: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.MechanicalDamping"),
    arguments: &[],
};

pub(super) fn undamped(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &UNDAMPED)
}

pub(super) const RAYLEIGH: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.MechanicalDamping"),
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.RayleighMassCoefficient")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.RayleighStiffnessCoefficient")]),
    ],
};

pub(super) fn rayleigh(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RAYLEIGH)
}
