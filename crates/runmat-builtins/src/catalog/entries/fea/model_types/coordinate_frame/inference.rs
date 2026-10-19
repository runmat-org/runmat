//! Static argument rules for `fea.CoordinateFrame` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const GLOBAL_FRAME: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_GLOBAL_COORDINATE_FRAME_CLASS,
    arguments: &[],
};

pub(super) fn global_frame(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &GLOBAL_FRAME)
}

pub(super) const MATERIAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MATERIAL_COORDINATE_FRAME_CLASS,
    arguments: &[],
};

pub(super) fn material(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MATERIAL)
}

pub(super) const ELEMENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELEMENT_COORDINATE_FRAME_CLASS,
    arguments: &[],
};

pub(super) fn element(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ELEMENT)
}

pub(super) const BOUNDARY_NORMAL_TANGENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_BOUNDARY_NORMAL_TANGENT_FRAME_CLASS,
    arguments: &[],
};

pub(super) fn boundary_normal_tangent(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &BOUNDARY_NORMAL_TANGENT)
}

pub(super) const CARTESIAN: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_CARTESIAN_COORDINATE_FRAME_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
    ],
};

pub(super) fn cartesian(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CARTESIAN)
}

pub(super) const CYLINDRICAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_CYLINDRICAL_COORDINATE_FRAME_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
    ],
};

pub(super) fn cylindrical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CYLINDRICAL)
}

pub(super) const SPHERICAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_SPHERICAL_COORDINATE_FRAME_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
        ModelArg::Typed(&[classes::FEA_DIRECTION_CLASS]),
    ],
};

pub(super) fn spherical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SPHERICAL)
}
