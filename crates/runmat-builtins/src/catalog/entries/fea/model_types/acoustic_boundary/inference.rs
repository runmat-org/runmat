//! Static argument rules for `fea.AcousticBoundary` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const RIGID_WALL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACOUSTIC_BOUNDARY_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn rigid_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RIGID_WALL)
}

pub(super) const RADIATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACOUSTIC_BOUNDARY_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn radiation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RADIATION)
}

pub(super) const IMPEDANCE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ACOUSTIC_BOUNDARY_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SpecificAcousticImpedance")]),
    ],
};

pub(super) fn impedance(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &IMPEDANCE)
}
