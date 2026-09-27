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
    let kind = if matches!(left.kind, ValueKindFact::Unknown)
        || matches!(right.kind, ValueKindFact::Unknown)
    {
        ValueKindFact::Unknown
    } else if symbolic {
        ValueKindFact::Symbolic
    } else {
        ValueKindFact::Logical
    };
    // Runtime text normalization treats a character operand as one string when
    // paired with string storage, not as a vector of numeric code points.
    let left_shape = if matches!(left.kind, ValueKindFact::Character)
        && matches!(right.kind, ValueKindFact::String)
    {
        ShapeFact::Scalar
    } else {
        left.shape.clone()
    };
    let right_shape = if matches!(right.kind, ValueKindFact::Character)
        && matches!(left.kind, ValueKindFact::String)
    {
        ShapeFact::Scalar
    } else {
        right.shape.clone()
    };
    let shape = match broadcast_shape(&left_shape, &right_shape) {
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
        residency: if left.residency == runmat_types::ResidencyFact::Host
            && right.residency == runmat_types::ResidencyFact::Host
        {
            runmat_types::ResidencyFact::Host
        } else {
            runmat_types::ResidencyFact::Unknown
        },
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: if matches!(left.kind, ValueKindFact::Unknown)
            || matches!(right.kind, ValueKindFact::Unknown)
        {
            runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        } else {
            runmat_types::CertaintyFact::Proven
        },
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

    #[test]
    fn unknown_relations_do_not_exclude_symbolic_outputs() {
        let entry = crate::builtin_catalog_entry_by_name("eq").unwrap();
        let mut input = numeric(ShapeFact::Scalar);
        input.residency = ResidencyFact::Device {
            provider: Some("test".into()),
        };
        let result = infer(
            &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue), input]),
            entry,
            RelationalOperator::Equal,
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    }

    #[test]
    fn character_string_comparison_uses_whole_text_not_character_shape() {
        let entry = crate::builtin_catalog_entry_by_name("eq").unwrap();
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(5)]),
            StorageFact::Dense,
        );
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        let result = infer(
            &request(vec![chars, strings.clone()]),
            entry,
            RelationalOperator::Equal,
        );
        assert_eq!(result.outputs[0].shape, strings.shape);
    }
}
