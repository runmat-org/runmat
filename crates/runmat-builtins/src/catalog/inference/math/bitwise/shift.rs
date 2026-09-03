use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
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
    if !matches!(value.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real)
    {
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
    }
    let shape = request.arguments.get(1).map_or_else(
        || value.shape.clone(),
        |count| scalar_or_equal(&value.shape, &count.shape),
    );
    let mut output = value.clone();
    output.shape = shape;
    output.storage = if output.shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Unknown
    };
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

fn scalar_or_equal(left: &ShapeFact, right: &ShapeFact) -> ShapeFact {
    if left.element_count() == Some(1) {
        right.clone()
    } else if right.element_count() == Some(1) || left == right {
        left.clone()
    } else {
        ShapeFact::Unknown
    }
}
