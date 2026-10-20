use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let count = request.arguments.len();
    let mut diagnostics = Vec::new();
    if !count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SPREADSHEET_IMPORT_OPTIONS-ARITY",
            "unsupported spreadsheetImportOptions argument count",
            0,
        ));
    }
    // Command forms and option-dependent output remain runtime-selected.
    finish_fixed_outputs(
        entry,
        request,
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        diagnostics,
    )
}
