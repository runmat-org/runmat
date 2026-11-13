//! Static argument rules for `fea.MagneticConstraint` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const VECTOR_POTENTIAL_GROUND: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn vector_potential_ground(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &VECTOR_POTENTIAL_GROUND)
}

pub(super) const COULOMB_GAUGE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn coulomb_gauge(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &COULOMB_GAUGE)
}

pub(super) const PRESCRIBED_VECTOR_POTENTIAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_CONSTRAINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.MagneticVectorPotential")]),
    ],
};

pub(super) fn prescribed_vector_potential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PRESCRIBED_VECTOR_POTENTIAL)
}

pub(super) const PERIODIC: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_CONSTRAINT_CLASS,
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

pub(super) const PHASE_SHIFTED_PERIODIC: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_CONSTRAINT_CLASS,
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
        ModelArg::Typed(&[StaticClassIdentity::new("fea.ComplexPhase")]),
    ],
};

pub(super) fn phase_shifted_periodic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &PHASE_SHIFTED_PERIODIC)
}
