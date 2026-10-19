//! Static argument rules for `fea.AcousticInitialCondition` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const PRESSURE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.AcousticInitialCondition"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[
            StaticClassIdentity::new("geometry.Region"),
            StaticClassIdentity::new("fea.InitialConditionTarget"),
        ]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.PressureSignal")]),
        ModelArg::RealScalar,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SignalSampleApproach")]),
        ModelArg::Text,
    ],
};

pub(super) fn pressure(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESSURE)
}
