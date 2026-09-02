use crate::{BuiltinCatalogEntry, ScalarLogicalReduction};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _reduction: ScalarLogicalReduction,
) -> CallInference {
    super::unary_logical_scalar::infer(request, entry, "RM-CATALOG-SCALAR-LOGICAL-REDUCTION-ARITY")
}
