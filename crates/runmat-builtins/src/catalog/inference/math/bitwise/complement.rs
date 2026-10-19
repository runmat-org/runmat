use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    diagnostics.extend(super::options::assumed_type(request, 1, &[0]));
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
    let kind = super::facts::result_kind(input);
    let supported = kind.is_some();
    let mut output = if let Some(kind) = kind {
        let mut output = ValueFact::proven(
            kind,
            input.shape.clone(),
            super::facts::storage(input, &input.shape),
        );
        if input.storage == StorageFact::Sparse {
            // Complementing implicit zero materializes a dense floating result;
            // the sparse mapper retains single precision where applicable.
            if matches!(input.kind, ValueKindFact::Numeric(_)) {
                output.kind = input.kind.clone();
            }
            output.storage = StorageFact::Dense;
        }
        output.residency = input.residency.clone();
        output
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
