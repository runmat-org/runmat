//! Static argument rules for `fea.ElectricalExcitation` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const TERMINAL_VOLTAGE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRICAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ElectricPotentialSignal")]),
    ],
};

pub(super) fn terminal_voltage(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &TERMINAL_VOLTAGE)
}

pub(super) const CURRENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRICAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ElectricCurrentSignal")]),
    ],
};

pub(super) fn current(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CURRENT)
}

pub(super) const CHARGE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRICAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ElectricChargeSignal")]),
    ],
};

pub(super) fn charge(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CHARGE)
}

pub(super) const CURRENT_DENSITY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRICAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.CurrentDensitySignal")]),
    ],
};

pub(super) fn current_density(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CURRENT_DENSITY)
}
