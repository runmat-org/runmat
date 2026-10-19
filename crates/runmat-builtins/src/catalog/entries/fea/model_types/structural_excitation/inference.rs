//! Static argument rules for `fea.StructuralExcitation` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const FORCE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ForceSignal")]),
    ],
};

pub(super) fn force(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &FORCE)
}

pub(super) const MOMENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.MomentSignal")]),
    ],
};

pub(super) fn moment(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MOMENT)
}

pub(super) const TRACTION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.TractionSignal")]),
    ],
};

pub(super) fn traction(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRACTION)
}

pub(super) const PRESSURE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.PressureSignal")]),
    ],
};

pub(super) fn pressure(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESSURE)
}

pub(super) const BODY_FORCE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.BodyForceSignal")]),
    ],
};

pub(super) fn body_force(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &BODY_FORCE)
}
