//! Static argument rules for `fea.StructuralMotion` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const ACCELERATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STRUCTURAL_TRANSLATION_MOTION_CLASS,
    arguments: &[
        ModelArg::Typed(&[classes::FEA_STRUCTURAL_TRANSLATION_DOF_CLASS]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.AccelerationSignal")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Displacement")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Velocity")]),
    ],
};

pub(super) fn acceleration(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ACCELERATION)
}
