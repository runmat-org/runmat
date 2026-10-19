//! Static argument rules for `fea.ReferencePoint` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const FROM_COORDINATES: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_REFERENCE_POINT_CLASS,
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Position")]),
    ],
};

pub(super) fn from_coordinates(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &FROM_COORDINATES)
}
