use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let shape = match request.arguments.len() {
        0 => ShapeFact::from(vec![Some(49), Some(49)]),
        1 => {
            let definitely_nonscalar = request.arguments[0].shape.is_definitely_nonscalar()
                || request
                    .literals
                    .numeric_vector_at(0)
                    .is_some_and(|vector| vector.len() > 1);
            if definitely_nonscalar {
                diagnostics.push(argument_error(
                    "RM-PEAKS-SCALAR-N",
                    "peaks grid size must be scalar",
                    0,
                ));
            }
            let n = if definitely_nonscalar {
                None
            } else {
                request.literals.numeric_dims().first().copied().flatten()
            };
            if n.is_some_and(|n| n <= 1) {
                diagnostics.push(argument_error(
                    "RM-PEAKS-N",
                    "peaks grid size must exceed one",
                    0,
                ));
            }
            ShapeFact::from(vec![n, n])
        }
        2 => {
            let left = &request.arguments[0].shape;
            let right = &request.arguments[1].shape;
            if left.is_proven_equivalent(right) {
                left.clone()
            } else {
                ShapeFact::Ranked { rank: 2 }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-PEAKS-ARITY",
                "peaks takes zero, one, or two inputs",
                2,
            ));
            ShapeFact::Unknown
        }
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    // A provider may select its own supported floating precision; host results
    // are double. Do not promise one physical class before placement.
    output.kind = ValueKindFact::Unknown;
    output.shape = shape;
    output.storage = StorageFact::Dense;
    finish_fixed_outputs(
        entry,
        request,
        vec![output.clone(), output.clone(), output],
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection};

    #[test]
    fn vector_grid_size_is_not_inferred_from_its_first_element() {
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Unknown,
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::new(vec![LiteralValue::Vector(vec![
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
            ])]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::PEAKS_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
    }
}
