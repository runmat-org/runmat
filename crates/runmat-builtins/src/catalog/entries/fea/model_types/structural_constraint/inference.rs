//! Static argument rules for `fea.StructuralConstraint` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const FIXED: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn fixed(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &FIXED)
}

pub(super) const PRESCRIBED: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::TypedOrCell(&[
            StaticClassIdentity::new("fea.StructuralTranslationDofValue"),
            StaticClassIdentity::new("fea.StructuralRotationDofValue"),
        ]),
        ModelArg::Typed(&[
            StaticClassIdentity::new("fea.GlobalCoordinateFrame"),
            StaticClassIdentity::new("fea.MaterialCoordinateFrame"),
            StaticClassIdentity::new("fea.ElementCoordinateFrame"),
            StaticClassIdentity::new("fea.BoundaryNormalTangentFrame"),
            StaticClassIdentity::new("fea.CartesianCoordinateFrame"),
            StaticClassIdentity::new("fea.CylindricalCoordinateFrame"),
            StaticClassIdentity::new("fea.SphericalCoordinateFrame"),
        ]),
    ],
};

pub(super) fn prescribed(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED)
}

pub(super) const PRESCRIBED_MOTION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::TypedOrCell(&[StaticClassIdentity::new("fea.StructuralTranslationMotion")]),
    ],
};

pub(super) fn prescribed_motion(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_MOTION)
}

pub(super) const SYMMETRY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn symmetry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SYMMETRY)
}

pub(super) const ANTISYMMETRY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn antisymmetry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ANTISYMMETRY)
}

pub(super) const NORMAL_DISPLACEMENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Displacement")]),
    ],
};

pub(super) fn normal_displacement(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &NORMAL_DISPLACEMENT)
}

pub(super) const RIGID_COUPLING: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ReferencePoint")]),
    ],
};

pub(super) fn rigid_coupling(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RIGID_COUPLING)
}

pub(super) const DISTRIBUTING_COUPLING: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ReferencePoint")]),
    ],
};

pub(super) fn distributing_coupling(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &DISTRIBUTING_COUPLING)
}

pub(super) const TIE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[
            StaticClassIdentity::new("fea.MatchingPeriodicPairing"),
            StaticClassIdentity::new("fea.GeometricPeriodicPairing"),
        ]),
    ],
};

pub(super) fn tie(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TIE)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[
            StaticClassIdentity::new("fea.IdentityPeriodicTransform"),
            StaticClassIdentity::new("fea.TranslationPeriodicTransform"),
            StaticClassIdentity::new("fea.RotationPeriodicTransform"),
        ]),
        ModelArg::Typed(&[
            StaticClassIdentity::new("fea.MatchingPeriodicPairing"),
            StaticClassIdentity::new("fea.GeometricPeriodicPairing"),
        ]),
    ],
};

pub(super) fn periodic(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PERIODIC)
}
