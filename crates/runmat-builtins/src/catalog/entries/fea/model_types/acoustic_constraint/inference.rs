//! Static argument rules for `fea.AcousticConstraint` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const PRESCRIBED_PRESSURE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.AcousticConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Pressure")]),
    ],
};

pub(super) fn prescribed_pressure(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_PRESSURE)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.AcousticConstraint"),
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

pub(super) const PHASE_SHIFTED_PERIODIC: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.AcousticConstraint"),
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
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ComplexPhase")]),
    ],
};

pub(super) fn phase_shifted_periodic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PHASE_SHIFTED_PERIODIC)
}
