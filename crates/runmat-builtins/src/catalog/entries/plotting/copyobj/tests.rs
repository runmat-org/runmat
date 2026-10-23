use super::*;
use runmat_types::*;
#[test]
fn canonical_output_contract_and_host_result() {
    let double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let arguments = vec![double.clone(), double.clone()];
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: arguments.clone(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &COPYOBJ_CATALOG_ENTRY);
        assert_eq!(result.diagnostics.is_empty(), count <= 1);
        assert_eq!(result.effects, COPYOBJ_CATALOG_ENTRY.contract.effect_set());
        if count == 1 {
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        }
    }
}

#[test]
fn equal_length_row_and_column_pairing_preserves_source_orientation() {
    let array = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        )
    };
    let source = ShapeFact::from(vec![Some(1), Some(2)]);
    let request = CallRequest {
        arguments: vec![
            array(source.clone()),
            array(ShapeFact::from(vec![Some(2), Some(1)])),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    };
    let result = inference::infer(&request, &COPYOBJ_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, source);
}
