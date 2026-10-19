//! Static argument rules for `fea.StructuralDofValue` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const TRANSLATION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralTranslationDofValue"),
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.StructuralTranslationDof")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Displacement")]),
    ],
};

pub(super) fn translation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSLATION)
}

pub(super) const ROTATION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralRotationDofValue"),
    arguments: &[
        ModelArg::Typed(&[StaticClassIdentity::new("fea.StructuralRotationDof")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Rotation")]),
    ],
};

pub(super) fn rotation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ROTATION)
}
