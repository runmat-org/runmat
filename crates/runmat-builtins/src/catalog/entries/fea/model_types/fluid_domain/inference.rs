//! Static argument rules for `fea.FluidDomain` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const NEWTONIAN: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidDomain"),
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.MassDensity")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.DynamicViscosity")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.TurbulenceIntensity")]),
    ],
};

pub(super) fn newtonian(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NEWTONIAN)
}
