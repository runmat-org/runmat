//! Static argument rules for `fea.ElectricalInitialCondition` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const POTENTIAL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalInitialCondition"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[
            StaticClassIdentity::new("geometry.Region"),
            StaticClassIdentity::new("fea.InitialConditionTarget"),
        ]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ElectricPotentialSignal")]),
        ModelArg::RealScalar,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.SignalSampleApproach")]),
        ModelArg::Text,
    ],
};

pub(super) fn potential(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &POTENTIAL)
}
