mod call;
mod class;

use crate::{BuiltinCatalogEntry, RemainderFunction};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    function: RemainderFunction,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    call::infer(function, request, entry)
}

#[cfg(test)]
mod tests;
