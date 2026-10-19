use super::super::transform_facts::{self, ArrayShapeTransform as Transform};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    transform_facts::infer(request, entry, Transform::Tile)
}
