//! Static argument rules for `fea.MagneticBoundary` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const INSULATION: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.MagneticBoundary"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
    ],
};

pub(super) fn insulation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &INSULATION)
}
