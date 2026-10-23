use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CLEAR_CACHE-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if crate::catalog::inference::known_invalid_scalar_object_handle(
            input,
            &[crate::MEMOIZED_FUNCTION_CLASS],
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CLEAR_CACHE-INPUT",
                "expected a scalar MemoizedFunction handle",
                0,
            ));
        }
    }
    let outputs = vec![];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
