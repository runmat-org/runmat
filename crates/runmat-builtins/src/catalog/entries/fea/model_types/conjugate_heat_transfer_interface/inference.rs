//! Static argument rules for `fea.ConjugateHeatTransferInterface` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const PERFECT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn perfect(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PERFECT)
}

pub(super) const CONDUCTANCE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.HeatTransferCoefficient")]),
    ],
};

pub(super) fn conductance(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CONDUCTANCE)
}

pub(super) const RESISTANCE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ThermalContactResistance")]),
    ],
};

pub(super) fn resistance(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RESISTANCE)
}
