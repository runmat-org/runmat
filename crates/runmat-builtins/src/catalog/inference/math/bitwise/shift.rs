use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    diagnostics.extend(super::options::numeric_control(request, 1));
    diagnostics.extend(super::options::assumed_type(request, 2, &[0]));
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BITSHIFT-ARITY",
            "bitshift requires a value and shift count and accepts one optional assumed type",
            request.arguments.len().min(2),
        ));
    }
    let Some(value) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let Some(kind) = super::facts::result_kind(value) else {
        if !matches!(value.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BITSHIFT-INPUT",
                "bitshift requires a real integer-valued numeric input",
                0,
            ));
        }
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    };
    let shape = request.arguments.get(1).map_or_else(
        || value.shape.clone(),
        |count| super::facts::scalar_or_equal(&value.shape, &count.shape),
    );
    let mut output = ValueFact::proven(kind, shape.clone(), super::facts::storage(value, &shape));
    output.residency = runmat_types::ResidencyFact::Unknown;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}
