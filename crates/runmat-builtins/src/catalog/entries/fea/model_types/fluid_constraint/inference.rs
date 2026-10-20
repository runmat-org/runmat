//! Static argument rules for `fea.FluidConstraint` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const NO_SLIP_WALL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn no_slip_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &NO_SLIP_WALL)
}

pub(super) const SLIP_WALL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn slip_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SLIP_WALL)
}

pub(super) const SYMMETRY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn symmetry(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &SYMMETRY)
}

pub(super) const PRESCRIBED_VELOCITY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::ObjectOrCellOf {
            object: &[StaticClassIdentity::new("fea.VelocitySignal")],
            cell: &[StaticClassIdentity::new("fea.Velocity")],
        },
    ],
};

pub(super) fn prescribed_velocity(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_VELOCITY)
}

pub(super) const MOVING_WALL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::ObjectOrCellOf {
            object: &[StaticClassIdentity::new("fea.VelocitySignal")],
            cell: &[StaticClassIdentity::new("fea.Velocity")],
        },
    ],
};

pub(super) fn moving_wall(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MOVING_WALL)
}

pub(super) const PRESSURE_REFERENCE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
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
    result: classes::FEA_FLUID_CONSTRAINT_CLASS,
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
