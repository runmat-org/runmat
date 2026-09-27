use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HOLD-ARITY",
            "hold accepts a mode with an optional leading axes handle",
            2,
        ));
    }
    if let Some((index, mode)) = request.arguments.iter().enumerate().next_back() {
        let invalid_kind = match mode.kind {
            ValueKindFact::Unknown | ValueKindFact::Character => false,
            ValueKindFact::String => {
                mode.storage != StorageFact::Scalar && mode.storage != StorageFact::Unknown
            }
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Complex
                    || mode.shape.element_count().is_some_and(|n| n != 1)
                    || mode.storage == StorageFact::Sparse
            }
            ValueKindFact::Logical => {
                mode.storage != StorageFact::Scalar && mode.storage != StorageFact::Unknown
            }
            _ => true,
        };
        let invalid_text = request
            .literals
            .literal_string_at(index)
            .is_some_and(|text| !matches!(text.trim(), "" | "on" | "off" | "all"));
        let invalid_number = request
            .literals
            .numeric_at(index)
            .is_some_and(|number| number != 0.0 && number != 1.0);
        if invalid_kind || invalid_text || invalid_number {
            diagnostics.push(argument_error("RM-CATALOG-HOLD-MODE", "hold requires on/off/all, empty text, or an exact numeric/logical scalar zero or one", index));
        }
    }
    if request.arguments.len() == 2 {
        let target = &request.arguments[0];
        if !matches!(
            target.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || target.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HOLD-TARGET",
                "hold target must be a scalar axes handle",
                0,
            ));
        }
    }
    // Handle existence and per-axes policy are runtime state; no value is returned.
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}
