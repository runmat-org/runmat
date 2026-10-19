//! Static argument rules for `fea.ExcitationCase` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const ALWAYS: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ExcitationCase"),
    arguments: &[
        ModelArg::Text,
        ModelArg::TypedOrCell(&[
            StaticClassIdentity::new("fea.StructuralExcitation"),
            StaticClassIdentity::new("fea.ThermalExcitation"),
            StaticClassIdentity::new("fea.ElectricalExcitation"),
            StaticClassIdentity::new("fea.MagneticExcitation"),
            StaticClassIdentity::new("fea.AcousticExcitation"),
            StaticClassIdentity::new("fea.FluidExcitation"),
        ]),
    ],
};

pub(super) fn always(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &ALWAYS)
}

pub(super) const DURING_STEPS: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ExcitationCase"),
    arguments: &[
        ModelArg::Text,
        ModelArg::TypedOrCell(&[
            StaticClassIdentity::new("fea.StructuralExcitation"),
            StaticClassIdentity::new("fea.ThermalExcitation"),
            StaticClassIdentity::new("fea.ElectricalExcitation"),
            StaticClassIdentity::new("fea.MagneticExcitation"),
            StaticClassIdentity::new("fea.AcousticExcitation"),
            StaticClassIdentity::new("fea.FluidExcitation"),
        ]),
        ModelArg::TextList,
    ],
};

pub(super) fn during_steps(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &DURING_STEPS)
}

pub(super) const MUTUALLY_EXCLUSIVE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ExcitationCase"),
    arguments: &[
        ModelArg::Text,
        ModelArg::TypedOrCell(&[
            StaticClassIdentity::new("fea.StructuralExcitation"),
            StaticClassIdentity::new("fea.ThermalExcitation"),
            StaticClassIdentity::new("fea.ElectricalExcitation"),
            StaticClassIdentity::new("fea.MagneticExcitation"),
            StaticClassIdentity::new("fea.AcousticExcitation"),
            StaticClassIdentity::new("fea.FluidExcitation"),
        ]),
        ModelArg::Text,
    ],
};

pub(super) fn mutually_exclusive(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &MUTUALLY_EXCLUSIVE)
}
