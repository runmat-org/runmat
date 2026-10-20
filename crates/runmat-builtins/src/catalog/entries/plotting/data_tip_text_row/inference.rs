use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let count = request.arguments.len();
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&count) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DATA_TIP-ARITY",
            "unsupported dataTipTextRow argument count",
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
