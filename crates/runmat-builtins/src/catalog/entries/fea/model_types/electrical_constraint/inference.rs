//! Static argument rules for `fea.ElectricalConstraint` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const EQUIPOTENTIAL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn equipotential(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &EQUIPOTENTIAL)
}

pub(super) const FLOATING_CONDUCTOR: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn floating_conductor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &FLOATING_CONDUCTOR)
}

pub(super) const POTENTIAL_REFERENCE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn potential_reference(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &POTENTIAL_REFERENCE)
}

pub(super) const PRESCRIBED_POTENTIAL: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ElectricPotential")]),
    ],
};

pub(super) fn prescribed_potential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_POTENTIAL)
}

pub(super) const INTEGRAL_CURRENT: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalConstraint"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ElectricCurrent")]),
    ],
};

pub(super) fn integral_current(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &INTEGRAL_CURRENT)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.ElectricalConstraint"),
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
