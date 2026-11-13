//! Static argument rules for `fea.FluidExcitation` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const INLET_VELOCITY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.VelocitySignal")]),
    ],
};

pub(super) fn inlet_velocity(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &INLET_VELOCITY)
}

pub(super) const MASS_FLOW: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.MassFlowSignal")]),
    ],
};

pub(super) fn mass_flow(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &MASS_FLOW)
}

pub(super) const WALL_MOTION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_FLUID_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.VelocitySignal")]),
    ],
};

pub(super) fn wall_motion(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &WALL_MOTION)
}
