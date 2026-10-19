//! Static argument rules for `fea.StructuralBoundary` constructors.

use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

pub(super) const PRESSURE: ModelObjectCall = ModelObjectCall {
    result: StaticClassIdentity::new("fea.StructuralBoundary"),
    arguments: &[
        ModelArg::Text,
        ModelArg::Typed(&[StaticClassIdentity::new("geometry.Region")]),
        ModelArg::Typed(&[StaticClassIdentity::new("fea.Pressure")]),
    ],
};

pub(super) fn pressure(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &PRESSURE)
}
