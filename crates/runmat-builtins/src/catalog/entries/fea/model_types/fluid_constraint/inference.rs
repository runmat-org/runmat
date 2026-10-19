//! Static argument rules for `fea.FluidConstraint` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const NO_SLIP_WALL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn no_slip_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NO_SLIP_WALL)
}

pub(super) const SLIP_WALL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn slip_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SLIP_WALL)
}

pub(super) const SYMMETRY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn symmetry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SYMMETRY)
}

pub(super) const PRESCRIBED_VELOCITY: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::TypedOrCell(&[StaticClassIdentity::new("fea.VelocitySignal")]),
    ],
};

pub(super) fn prescribed_velocity(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_VELOCITY)
}

pub(super) const MOVING_WALL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::TypedOrCell(&[StaticClassIdentity::new("fea.VelocitySignal")]),
    ],
};

pub(super) fn moving_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MOVING_WALL)
}

pub(super) const PRESSURE_REFERENCE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Pressure")]),
    ],
};

pub(super) fn pressure_reference(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESSURE_REFERENCE)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.FluidConstraint"),
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
