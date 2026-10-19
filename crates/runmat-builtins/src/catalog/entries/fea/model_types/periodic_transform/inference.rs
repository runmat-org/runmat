//! Static argument rules for `fea.PeriodicTransform` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const IDENTITY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.IdentityPeriodicTransform"),
    arguments: &[],
};

pub(super) fn identity(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &IDENTITY)
}

pub(super) const TRANSLATION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.TranslationPeriodicTransform"),
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "fea.Translation",
    )])],
};

pub(super) fn translation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSLATION)
}

pub(super) const ROTATION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.RotationPeriodicTransform"),
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Rotation")]),
    ],
};

pub(super) fn rotation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ROTATION)
}
