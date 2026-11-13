//! Static argument rules for `fea.ThermalExcitation` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const HEAT_GENERATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.HeatGenerationSignal")]),
    ],
};

pub(super) fn heat_generation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &HEAT_GENERATION)
}

pub(super) const HEAT_FLUX: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.HeatFluxSignal")]),
    ],
};

pub(super) fn heat_flux(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &HEAT_FLUX)
}

pub(super) const TEMPERATURE_FIELD: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.TemperatureSignal")]),
    ],
};

pub(super) fn temperature_field(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &TEMPERATURE_FIELD)
}

pub(super) const AMBIENT_TEMPERATURE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.TemperatureSignal")]),
    ],
};

pub(super) fn ambient_temperature(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &AMBIENT_TEMPERATURE)
}
