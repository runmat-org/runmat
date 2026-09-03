use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, NumericDomain, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BITCMP-ARITY",
            "bitcmp requires one input and accepts one optional assumed type",
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
    let supported = matches!(input.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real);
    let mut output = if supported {
        input.clone()
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    if !supported && !matches!(input.kind, ValueKindFact::Unknown) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BITCMP-INPUT",
            "bitcmp requires a real integer-valued numeric input",
            0,
        ));
    }
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}
