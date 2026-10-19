//! Static argument rules for `fea.StructuralDof` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const UX: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralTranslationDof"),
    arguments: &[],
};

pub(super) fn ux(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &UX)
}

pub(super) const UY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralTranslationDof"),
    arguments: &[],
};

pub(super) fn uy(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &UY)
}

pub(super) const UZ: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralTranslationDof"),
    arguments: &[],
};

pub(super) fn uz(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &UZ)
}

pub(super) const RX: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralRotationDof"),
    arguments: &[],
};

pub(super) fn rx(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RX)
}

pub(super) const RY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralRotationDof"),
    arguments: &[],
};

pub(super) fn ry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RY)
}

pub(super) const RZ: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralRotationDof"),
    arguments: &[],
};

pub(super) fn rz(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RZ)
}
