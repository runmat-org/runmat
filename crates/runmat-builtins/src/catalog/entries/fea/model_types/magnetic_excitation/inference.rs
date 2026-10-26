//! Static argument rules for `fea.MagneticExcitation` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const VECTOR_POTENTIAL: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_MAGNETIC_EXCITATION_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new(
            "fea.MagneticVectorPotentialSignal",
        )]),
    ],
};

pub(super) fn vector_potential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &VECTOR_POTENTIAL)
}
