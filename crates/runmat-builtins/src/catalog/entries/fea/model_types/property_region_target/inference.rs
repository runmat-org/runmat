//! Static argument rules for `fea.PropertyRegionTarget` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const BEAM: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Region",
    )])],
};

pub(super) fn beam(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &BEAM)
}

pub(super) const SHELL: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Region",
    )])],
};

pub(super) fn shell(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SHELL)
}

pub(super) const SOLID: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Region",
    )])],
};

pub(super) fn solid(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SOLID)
}

pub(super) const CELL: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Region",
    )])],
};

pub(super) fn cell(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CELL)
}

pub(super) const ALL_BEAMS: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Asset",
    )])],
};

pub(super) fn all_beams(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ALL_BEAMS)
}

pub(super) const ALL_SHELLS: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Asset",
    )])],
};

pub(super) fn all_shells(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ALL_SHELLS)
}

pub(super) const ALL_SOLIDS: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Asset",
    )])],
};

pub(super) fn all_solids(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ALL_SOLIDS)
}

pub(super) const ALL_CELLS: ModelObjectCall = ModelObjectCall {
    result: crate::fea_contract::identities::FEA_PROPERTY_REGION_TARGET_CLASS,
    arguments: &[ModelArg::Typed(&[StaticClassIdentity::new(
        "geometry.Asset",
    )])],
};

pub(super) fn all_cells(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ALL_CELLS)
}
