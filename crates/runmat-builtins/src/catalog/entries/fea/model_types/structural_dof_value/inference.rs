//! Static argument rules for `fea.StructuralDofValue` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const TRANSLATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_TRANSLATION_DOF_VALUE_CLASS,
    arguments: &[
        ModelArg::Typed(&[classes::FEA_STRUCTURAL_TRANSLATION_DOF_CLASS]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Displacement")]),
    ],
};

pub(super) fn translation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TRANSLATION)
}

pub(super) const ROTATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_ROTATION_DOF_VALUE_CLASS,
    arguments: &[
        ModelArg::Typed(&[classes::FEA_STRUCTURAL_ROTATION_DOF_CLASS]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Rotation")]),
    ],
};

pub(super) fn rotation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ROTATION)
}
