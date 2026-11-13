//! Static argument rules for `fea.ElectricalConstraint` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const EQUIPOTENTIAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRICAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn equipotential(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &EQUIPOTENTIAL)
}

pub(super) const FLOATING_CONDUCTOR: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_ELECTRICAL_CONSTRAINT_CLASS,
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
    result: classes::FEA_ELECTRICAL_CONSTRAINT_CLASS,
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
    result: classes::FEA_ELECTRICAL_CONSTRAINT_CLASS,
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
    result: classes::FEA_ELECTRICAL_CONSTRAINT_CLASS,
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
    result: classes::FEA_ELECTRICAL_CONSTRAINT_CLASS,
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
