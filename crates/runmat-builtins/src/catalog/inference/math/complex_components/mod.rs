mod call;

use crate::{BuiltinCatalogEntry, NumericComponentRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rule: NumericComponentRule,
) -> CallInference {
    call::infer_numeric_component_call(request, entry, rule)
}

#[cfg(test)]
mod tests;
