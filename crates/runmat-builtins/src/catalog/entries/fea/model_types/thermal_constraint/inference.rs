//! Static argument rules for `fea.ThermalConstraint` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const PRESCRIBED: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ThermalConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Temperature")]),
    ],
};

pub(super) fn prescribed(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED)
}

pub(super) const TIE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ThermalConstraint"),
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
    result: StaticClassIdentity::new("fea.ThermalConstraint"),
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

pub(super) const AFFINE_JUMP: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ThermalConstraint"),
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
        ModelArg::Typed(&[StaticClassIdentity::new("fea.TemperatureDifference")]),
    ],
};

pub(super) fn affine_jump(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &AFFINE_JUMP)
}
