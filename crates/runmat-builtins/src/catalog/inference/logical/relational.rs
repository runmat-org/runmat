use crate::{BuiltinCatalogEntry, RelationalOperator};
use runmat_types::{
    broadcast_shape, AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason,
    LayoutFact, MutationFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _operator: RelationalOperator,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(super::super::argument_error(
            "RM-CATALOG-RELATIONAL-ARITY",
            format!("{} requires exactly two inputs", entry.identity.name),
            request.arguments.len().min(1),
        ));
    }
    let (Some(left), Some(right)) = (request.arguments.first(), request.arguments.get(1)) else {
        return super::super::finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let symbolic = matches!(left.kind, ValueKindFact::Symbolic)
        || matches!(right.kind, ValueKindFact::Symbolic);
    let kind = if symbolic {
        ValueKindFact::Symbolic
    } else {
        ValueKindFact::Logical
    };
    let shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let output = ValueFact {
        kind,
        shape,
        storage,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: super::super::preserved_binary_residency(&left.residency, &right.residency),
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: runmat_types::CertaintyFact::Proven,
        invalidation: Default::default(),
    };
    super::super::finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    fn numeric(shape: ShapeFact) -> ValueFact {
        let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        fact.shape = shape;
        fact.storage = StorageFact::Dense;
        fact
    }

    #[test]
    fn relational_inference_broadcasts_to_logical() {
        let entry = crate::builtin_catalog_entry_by_name("lt").expect("lt entry");
        let request = request(vec![
            numeric(ShapeFact::from(vec![Some(3), Some(1)])),
            numeric(ShapeFact::from(vec![Some(1), Some(4)])),
        ]);
        let inference = infer(&request, entry, RelationalOperator::LessThan);
        assert!(inference.diagnostics.is_empty());
        assert_eq!(inference.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(
            inference.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(4)])
        );
    }

    #[test]
    fn symbolic_relations_preserve_symbolic_results() {
        let entry = crate::builtin_catalog_entry_by_name("ge").expect("ge entry");
        let request = request(vec![
            ValueFact::scalar(ValueKindFact::Symbolic),
            numeric(ShapeFact::Scalar),
        ]);
        let inference = infer(&request, entry, RelationalOperator::GreaterThanOrEqual);
        assert!(inference.diagnostics.is_empty());
        assert_eq!(inference.outputs[0].kind, ValueKindFact::Symbolic);
        assert_eq!(inference.outputs[0].residency, ResidencyFact::Host);
    }
}
