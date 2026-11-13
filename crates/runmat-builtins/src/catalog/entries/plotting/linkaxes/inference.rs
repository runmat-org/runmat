use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINKAXES-ARITY",
            "linkaxes requires handles and an optional link mode",
            0,
        ));
    }
    if let Some(handles) = request.arguments.first() {
        if !matches!(
            handles.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || handles
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Complex)
            || handles.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAXES-HANDLES",
                "linkaxes requires real host numeric axes handles",
                0,
            ));
        }
        // All ranks, empty arrays and duplicates are accepted by the existing
        // domain. Existence and deduplication belong to the shared state owner.
    }
    if let Some(mode) = request.arguments.get(1) {
        let invalid_kind = !matches!(
            mode.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        );
        let invalid_literal = request
            .literals
            .literal_string_at(1)
            .is_some_and(|s| !matches!(s.trim(), "x" | "y" | "xy" | "off"));
        if invalid_kind || invalid_literal {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAXES-MODE",
                "linkaxes mode must be x, y, xy or off",
                1,
            ));
        }
    }
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}
