use crate::{BuiltinCatalogEntry, ShapePredicate};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _predicate: ShapePredicate,
) -> CallInference {
    super::super::super::unary_logical_scalar::infer(
        request,
        entry,
        "RM-CATALOG-SHAPE-PREDICATE-ARITY",
    )
}
