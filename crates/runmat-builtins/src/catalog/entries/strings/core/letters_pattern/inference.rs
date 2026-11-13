use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        super::super::pattern::facts::pattern_fact(),
        super::super::pattern::facts::bounded_count_diagnostics(request, "lettersPattern", false),
    )
}
