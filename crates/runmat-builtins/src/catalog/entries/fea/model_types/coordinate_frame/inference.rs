//! Static argument rules for `fea.CoordinateFrame` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const GLOBAL_FRAME: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.GlobalCoordinateFrame"),
    arguments: &[],
};

pub(super) fn global_frame(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &GLOBAL_FRAME)
}

pub(super) const MATERIAL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.MaterialCoordinateFrame"),
    arguments: &[],
};

pub(super) fn material(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MATERIAL)
}

pub(super) const ELEMENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElementCoordinateFrame"),
    arguments: &[],
};

pub(super) fn element(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ELEMENT)
}

pub(super) const BOUNDARY_NORMAL_TANGENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.BoundaryNormalTangentFrame"),
    arguments: &[],
};

pub(super) fn boundary_normal_tangent(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &BOUNDARY_NORMAL_TANGENT)
}

pub(super) const CARTESIAN: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.CartesianCoordinateFrame"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
    ],
};

pub(super) fn cartesian(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CARTESIAN)
}

pub(super) const CYLINDRICAL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.CylindricalCoordinateFrame"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
    ],
};

pub(super) fn cylindrical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CYLINDRICAL)
}

pub(super) const SPHERICAL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.SphericalCoordinateFrame"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Direction")]),
    ],
};

pub(super) fn spherical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SPHERICAL)
}
