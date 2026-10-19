//! Static argument rules for `fea.MagneticInitialCondition` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const VECTOR_POTENTIAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_INITIAL_CONDITION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[
            StaticClassIdentity::new("geometry.Region"),
            classes::FEA_INITIAL_CONDITION_TARGET_CLASS,
        ]),
        ModelArg::Typed(&[StaticClassIdentity::new(
            "fea.MagneticVectorPotentialSignal",
        )]),
        ModelArg::RealScalar,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SignalSampleApproach")]),
        ModelArg::Text,
    ],
};

pub(super) fn vector_potential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &VECTOR_POTENTIAL)
}
