//! Static argument rules for `fea.ThermalConstraint` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const PRESCRIBED: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Temperature")]),
    ],
};

pub(super) fn prescribed(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED)
}

pub(super) const TIE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[
            classes::FEA_MATCHING_PERIODIC_PAIRING_CLASS,
            classes::FEA_GEOMETRIC_PERIODIC_PAIRING_CLASS,
        ]),
    ],
};

pub(super) fn tie(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &TIE)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_CONSTRAINT_CLASS,
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

pub(super) const AFFINE_JUMP: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_THERMAL_CONSTRAINT_CLASS,
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
        ModelArg::Typed(&[StaticClassIdentity::new("fea.TemperatureDifference")]),
    ],
};

pub(super) fn affine_jump(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &AFFINE_JUMP)
}
