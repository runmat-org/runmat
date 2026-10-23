use super::*;
use runmat_types::*;
#[test]
fn canonical_output_contract_and_host_result() {
    let arguments = vec![];
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: arguments.clone(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &GOBJECTS_CATALOG_ENTRY);
        assert_eq!(result.diagnostics.is_empty(), count <= 1);
        assert_eq!(result.effects, GOBJECTS_CATALOG_ENTRY.contract.effect_set());
        if count == 1 {
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        }
    }
}

#[test]
fn dimensions_preserve_empty_and_native_integer_literals() {
    for (literal, expected) in [
        (LiteralValue::Number(0.0), vec![Some(0), Some(0)]),
        (
            LiteralValue::Integer {
                text: "-2".into(),
                class: NumericClass::Int64,
            },
            vec![Some(0), Some(0)],
        ),
        (
            LiteralValue::Vector(vec![
                LiteralValue::Number(2.0),
                LiteralValue::Number(0.0),
                LiteralValue::Number(3.0),
            ]),
            vec![Some(2), Some(0), Some(3)],
        ),
    ] {
        let input = runmat_types::infer_literal(&literal).fact;
        let request = CallRequest {
            arguments: vec![input],
            literals: LiteralContext {
                literal_args: vec![literal],
            },
            outputs: OutputSelection::default(),
        };
        let result = inference::infer(&request, &GOBJECTS_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, ShapeFact::from(expected));
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
    }
}
