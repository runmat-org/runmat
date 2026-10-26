//! Static argument rules for `fea.StructuralInitialCondition` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const DISPLACEMENT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_INITIAL_CONDITION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[
            StaticClassIdentity::new("geometry.Region"),
            classes::FEA_INITIAL_CONDITION_TARGET_CLASS,
        ]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.DisplacementSignal")]),
        ModelArg::RealScalar,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SignalSampleApproach")]),
        ModelArg::Text,
    ],
};

pub(super) fn displacement(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &DISPLACEMENT)
}

pub(super) const VELOCITY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_INITIAL_CONDITION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[
            StaticClassIdentity::new("geometry.Region"),
            classes::FEA_INITIAL_CONDITION_TARGET_CLASS,
        ]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.VelocitySignal")]),
        ModelArg::RealScalar,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SignalSampleApproach")]),
        ModelArg::Text,
    ],
};

pub(super) fn velocity(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &VELOCITY)
}

pub(super) const ANGULAR_VELOCITY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_INITIAL_CONDITION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[
            StaticClassIdentity::new("geometry.Region"),
            classes::FEA_INITIAL_CONDITION_TARGET_CLASS,
        ]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.AngularVelocitySignal")]),
        ModelArg::RealScalar,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SignalSampleApproach")]),
        ModelArg::Text,
    ],
};

pub(super) fn angular_velocity(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &ANGULAR_VELOCITY)
}
