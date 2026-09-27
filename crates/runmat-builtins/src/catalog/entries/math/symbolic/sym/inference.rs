use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SYM-ARITY",
            "sym requires one input and does not implement options",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        let valid_kind = matches!(
            input.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
                | ValueKindFact::Logical
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Symbolic
        );
        let invalid_shape = if input.kind == ValueKindFact::Character {
            input.shape.known_dims().is_some_and(|dims| {
                dims.first().is_some_and(|d| d.is_some_and(|d| d != 1))
                    || dims.iter().skip(2).any(|d| d.is_some_and(|d| d != 1))
            })
        } else if input.kind != ValueKindFact::Unknown {
            input.shape.element_count().is_some_and(|n| n != 1)
        } else {
            false
        };
        if !valid_kind
            || invalid_shape
            || (input.kind == ValueKindFact::Logical && input.storage == StorageFact::Dense)
            || input.storage == StorageFact::Sparse
            || matches!(input.residency, ResidencyFact::Device { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SYM-INPUT",
                "sym requires host scalar symbolic, real numeric, logical, or scalar text input",
                0,
            ));
        }
    }
    // Success always constructs a host symbolic scalar, even for unknown input.
    // No numeric literal is converted to f64 to establish this result.
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Symbolic),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_scalar_and_array_representations_match_runtime_admission() {
        for (kind, storage, rejected) in [
            (ValueKindFact::Logical, StorageFact::Scalar, false),
            (ValueKindFact::Logical, StorageFact::Dense, true),
            (ValueKindFact::Logical, StorageFact::Unknown, false),
            (ValueKindFact::Unknown, StorageFact::Dense, false),
        ] {
            let request = CallRequest {
                arguments: vec![ValueFact::proven(kind, ShapeFact::Scalar, storage)],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            };
            let result = infer(&request, &super::super::SYM_CATALOG_ENTRY);
            assert_eq!(!result.diagnostics.is_empty(), rejected);
        }
    }
    #[test]
    fn unknown_and_exact_integer_literals_have_symbolic_scalar_result() {
        for literal in [
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "18446744073709551615".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Character("9007199254740993/7".into()),
        ] {
            let request = CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                literals: LiteralContext::new(vec![literal]),
                outputs: OutputSelection::default(),
            };
            let result = infer(&request, &super::super::SYM_CATALOG_ENTRY);
            assert!(result.diagnostics.is_empty());
            assert_eq!(
                result.outputs[0],
                ValueFact::scalar(ValueKindFact::Symbolic)
            );
        }
    }
    #[test]
    fn rejects_known_array_and_excess_outputs() {
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::from(vec![Some(2), Some(3)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        assert!(
            infer(&request, &super::super::SYM_CATALOG_ENTRY)
                .diagnostics
                .len()
                >= 2
        );
    }
}
