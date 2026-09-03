use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, NumericDomain,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-SWAPBYTES-ARITY",
            "swapbytes requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SWAPBYTES-ARITY",
            "swapbytes accepts exactly one input",
            1,
        ));
    }

    let mut output = input.clone();
    let supported = matches!(output.kind, ValueKindFact::Numeric(ref numeric) if numeric.domain == NumericDomain::Real)
        && !matches!(output.storage, StorageFact::Sparse);
    if supported {
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
        output.residency = ResidencyFact::Host;
    } else if matches!(output.kind, ValueKindFact::Unknown) {
        output = ValueFact::unknown(DynamicReason::RuntimeValue);
    } else {
        diagnostics.push(argument_error(
            "RM-CATALOG-SWAPBYTES-INPUT",
            "swapbytes requires a real dense numeric scalar or array",
            0,
        ));
        output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    finish_fixed(entry, request, output, diagnostics)
}
