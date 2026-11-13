use super::*;
use runmat_types::*;

fn real(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &BUTTORD_CATALOG_ENTRY,
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn scalar_and_pair_edges_produce_host_double_outputs() {
    for (shape, cutoff) in [
        (ShapeFact::Scalar, ShapeFact::Scalar),
        (
            ShapeFact::from(vec![Some(1), Some(2)]),
            ShapeFact::from(vec![Some(1), Some(2)]),
        ),
    ] {
        let result = infer(
            vec![
                real(shape.clone()),
                real(shape),
                real(ShapeFact::Scalar),
                real(ShapeFact::Scalar),
            ],
            vec![LiteralValue::Unknown; 4],
            2,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[1].shape, cutoff);
        for output in &result.outputs {
            assert_eq!(
                output.numeric().map(|n| n.class),
                Some(NumericClass::Double)
            );
            assert_eq!(output.residency, ResidencyFact::Host);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invalid_known_control_and_edges_do_not_claim_success_facts() {
    let edge = real(ShapeFact::Scalar);
    let attenuation = real(ShapeFact::Scalar);
    let mut args = vec![edge.clone(), edge, attenuation.clone(), attenuation];
    let bad = infer(
        args.clone(),
        vec![
            LiteralValue::Number(0.0),
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ],
        2,
    );
    assert!(!bad.diagnostics.is_empty());
    assert_eq!(bad.outputs[0].kind, ValueKindFact::Unknown);
    args.push(ValueFact::unknown(DynamicReason::RuntimeValue));
    let bad = infer(
        args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Character("digital".into()),
        ],
        2,
    );
    assert!(!bad.diagnostics.is_empty());
    assert_eq!(bad.outputs[1].kind, ValueKindFact::Unknown);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unknown_edges_leave_cutoff_shape_dynamic_and_extensions_remain_distinct() {
    let mut args = vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2];
    args.extend([real(ShapeFact::Scalar), real(ShapeFact::Scalar)]);
    let result = infer(args, vec![LiteralValue::Unknown; 4], 2);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
    assert_eq!(BUTTORD_CATALOG_ENTRY.extensions.len(), 6);
    assert_eq!(BUTTORD_CATALOG_ENTRY.integer_capabilities.len(), 2);
}
