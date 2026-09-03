use super::common::finish_materialized_real;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMMA-ARITY",
            "gamma requires exactly one input",
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let ValueKindFact::Numeric(numeric) = input.kind else {
        let reason = if matches!(input.kind, ValueKindFact::Unknown) {
            DynamicReason::RuntimeValue
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAMMA-INPUT",
                "gamma requires real single or double input",
                0,
            ));
            DynamicReason::UnsupportedRepresentation
        };
        return finish_fixed(entry, request, ValueFact::unknown(reason), diagnostics);
    };
    if numeric.domain != NumericDomain::Real
        || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        || matches!(input.storage, StorageFact::Sparse)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMMA-INPUT",
            "gamma requires dense real single or double input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    finish_materialized_real(entry, request, input, numeric.class, diagnostics)
}
