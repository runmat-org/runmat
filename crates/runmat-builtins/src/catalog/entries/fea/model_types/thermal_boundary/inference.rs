//! Static argument rules for `fea.ThermalBoundary` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const CONVECTION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_BOUNDARY_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Temperature")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.HeatTransferCoefficient")]),
    ],
};

pub(super) fn convection(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CONVECTION)
}

pub(super) const HEAT_FLUX: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_BOUNDARY_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.HeatFlux")]),
    ],
};

pub(super) fn heat_flux(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &HEAT_FLUX)
}
