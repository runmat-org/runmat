use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_concatenate, CallInference, CallRequest, CertaintyFact, DynamicReason, FactJoin,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn horizontal(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, 2)
}

pub(super) fn vertical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, 1)
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, dimension: usize) -> CallInference {
    let (output, diagnostics) = match request.arguments.as_slice() {
        [] => {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = ShapeFact::from(vec![Some(0), Some(0)]);
            output.residency = ResidencyFact::Host;
            (output, Vec::new())
        }
        [only] => (only.clone(), Vec::new()),
        inputs if admits_shape_projection(inputs) => {
            let result = infer_concatenate(dimension, inputs);
            let mut output = result.fact;
            if result.diagnostics.is_empty()
                && inputs
                    .iter()
                    .all(|input| matches!(&input.kind, ValueKindFact::Cell(_)))
            {
                let joined = inputs
                    .iter()
                    .map(|input| input.kind.clone())
                    .reduce(|left, right| left.join(&right));
                if let Some(ValueKindFact::Cell(mut cell)) = joined {
                    // Shared element summaries survive. Old linear positions do
                    // not: concatenation changes their column-major order.
                    cell.elements.clear();
                    cell.elements_complete = false;
                    output.kind = ValueKindFact::Cell(cell);
                    output.storage = StorageFact::Dense;
                }
            }
            output.invalidation = inputs
                .iter()
                .map(|input| input.invalidation.clone())
                .reduce(|left, right| left.join(&right))
                .unwrap_or_default();
            output.certainty = inputs
                .iter()
                .map(|input| input.certainty.clone())
                .reduce(|left, right| left.join(&right))
                .unwrap_or(CertaintyFact::Dynamic(DynamicReason::RuntimeValue));
            if matches!(&output.kind, ValueKindFact::Unknown) {
                output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
            }
            (output, result.diagnostics)
        }
        _ => (ValueFact::unknown(DynamicReason::RuntimeValue), Vec::new()),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn admits_shape_projection(inputs: &[ValueFact]) -> bool {
    let host_nonempty_dense = inputs.iter().all(|input| {
        input.residency == ResidencyFact::Host
            && input.shape.element_count().is_some_and(|count| count > 0)
            && matches!(input.storage, StorageFact::Scalar | StorageFact::Dense)
    });
    if !host_nonempty_dense {
        return false;
    }
    let all_numeric_or_logical = inputs.iter().all(|input| {
        matches!(
            &input.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    });
    let all_character = inputs
        .iter()
        .all(|input| matches!(&input.kind, ValueKindFact::Character));
    let all_cell = inputs
        .iter()
        .all(|input| matches!(&input.kind, ValueKindFact::Cell(_)));
    all_numeric_or_logical || all_character || all_cell
}
