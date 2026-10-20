//! Static argument rules for `fea.StructuralConstraint` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const FIXED: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn fixed(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &FIXED)
}

pub(super) const PRESCRIBED: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::TypedOrCell(&[
            classes::FEA_STRUCTURAL_TRANSLATION_DOF_VALUE_CLASS,
            classes::FEA_STRUCTURAL_ROTATION_DOF_VALUE_CLASS,
        ]),
        ModelArg::Typed(classes::FEA_COORDINATE_FRAME_CLASSES),
    ],
};

pub(super) fn prescribed(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED)
}

pub(super) const PRESCRIBED_MOTION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::TypedOrCell(&[classes::FEA_STRUCTURAL_TRANSLATION_MOTION_CLASS]),
    ],
};

pub(super) fn prescribed_motion(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_MOTION)
}

pub(super) const SYMMETRY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn symmetry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SYMMETRY)
}

pub(super) const ANTISYMMETRY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn antisymmetry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ANTISYMMETRY)
}

pub(super) const NORMAL_DISPLACEMENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
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
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[classes::FEA_REFERENCE_POINT_CLASS]),
    ],
};

pub(super) fn rigid_coupling(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &RIGID_COUPLING)
}

pub(super) const DISTRIBUTING_COUPLING: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[classes::FEA_REFERENCE_POINT_CLASS]),
    ],
};

pub(super) fn distributing_coupling(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &DISTRIBUTING_COUPLING)
}

pub(super) const TIE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[
            classes::FEA_MATCHING_PERIODIC_PAIRING_CLASS,
            classes::FEA_GEOMETRIC_PERIODIC_PAIRING_CLASS,
        ]),
    ],
};

pub(super) fn tie(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TIE)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[
            classes::FEA_IDENTITY_PERIODIC_TRANSFORM_CLASS,
            classes::FEA_TRANSLATION_PERIODIC_TRANSFORM_CLASS,
            classes::FEA_ROTATION_PERIODIC_TRANSFORM_CLASS,
        ]),
        ModelArg::Typed(&[
            classes::FEA_MATCHING_PERIODIC_PAIRING_CLASS,
            classes::FEA_GEOMETRIC_PERIODIC_PAIRING_CLASS,
        ]),
    ],
};

pub(super) fn periodic(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PERIODIC)
}
