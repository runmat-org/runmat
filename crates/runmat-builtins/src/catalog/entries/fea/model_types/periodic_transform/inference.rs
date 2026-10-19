//! Static argument rules for `fea.PeriodicTransform` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const IDENTITY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_IDENTITY_PERIODIC_TRANSFORM_CLASS,
    arguments: &[],
};

pub(super) fn identity(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &IDENTITY)
}

pub(super) const TRANSLATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_TRANSLATION_PERIODIC_TRANSFORM_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "fea.Translation",
    )])],
};

pub(super) fn translation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSLATION)
}

pub(super) const ROTATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ROTATION_PERIODIC_TRANSFORM_CLASS,
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Rotation")]),
    ],
};

pub(super) fn rotation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ROTATION)
}
