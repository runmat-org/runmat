use crate::{BuiltinCatalogEntry, MetadataPredicate};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _predicate: MetadataPredicate,
) -> CallInference {
    super::unary_logical_scalar::infer(request, entry, "RM-CATALOG-METADATA-PREDICATE-ARITY")
}
