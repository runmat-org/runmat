use super::*;
use runmat_types::*;

fn request(input: ValueFact, count: usize) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn scalar_unknown_and_discarded_forms_preserve_logical_result_and_effects() {
    for input in [
        ValueFact::unknown(DynamicReason::RuntimeValue),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ] {
        let inferred = inference::infer(&request(input.clone(), 1), &FEOF_CATALOG_ENTRY);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs.len(), 1);
        assert_eq!(
            inferred.outputs[0],
            ValueFact::scalar(ValueKindFact::Logical)
        );
        let discarded = inference::infer(&request(input, 0), &FEOF_CATALOG_ENTRY);
        assert!(discarded.outputs.is_empty());
        assert!(discarded.effects.0.contains(&EffectKind::EnvironmentRead));
    }
}

#[test]
fn known_wrong_class_geometry_and_output_count_are_diagnosed() {
    for input in [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
    ] {
        assert!(!inference::infer(&request(input, 1), &FEOF_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
    let too_many = inference::infer(
        &request(ValueFact::unknown(DynamicReason::RuntimeValue), 2),
        &FEOF_CATALOG_ENTRY,
    );
    assert!(!too_many.diagnostics.is_empty());
}
