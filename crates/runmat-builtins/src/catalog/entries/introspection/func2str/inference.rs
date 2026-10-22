use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FUNC2STR-ARITY",
            "expected one function handle",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Callable(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FUNC2STR-HANDLE",
                "expected a function handle",
                0,
            ));
        }
        if matches!(
            input.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FUNC2STR-RESIDENCY",
                "function handle must be host-resident",
                0,
            ));
        }
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![ValueFact::scalar(ValueKindFact::String)],
        diagnostics,
    )
}
