use super::*;
use runmat_types::*;
fn request(outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(4), Some(2)]),
            StorageFact::Dense,
        )],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn complete_contract_has_real_output_schema_and_closed_options() {
    let result = inference::infer(&request(4), &TTEST_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 4);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Logical));
    assert!(matches!(
        result.outputs[1].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
    let ValueKindFact::Struct(stats) = &result.outputs[3].kind else {
        panic!("stats")
    };
    assert!(stats.fields_complete);
    assert_eq!(stats.fields.len(), 3);
    assert!(
        stats.fields.contains_key("tstat")
            && stats.fields.contains_key("df")
            && stats.fields.contains_key("sd")
    );
    assert_eq!(
        TtestOptionName::from_public("aLpHa"),
        Some(TtestOptionName::Alpha)
    );
    assert_eq!(TtestOptionName::from_public("VarType"), None);
    assert_eq!(TtestTail::from_public("larger"), None);
    assert!(TTEST_CATALOG_ENTRY.integer_capabilities.is_empty());
    assert_eq!(TTEST_CATALOG_ENTRY.documentation.examples.len(), 2);
    assert!(!inference::infer(&request(5), &TTEST_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn known_dimension_and_dynamic_option_names_have_sound_distinct_precision() {
    let mut request = request(4);
    request.arguments.extend([
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ]);
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Character("Dim".into()),
        LiteralValue::Number(2.0),
    ]);
    let result = inference::infer(&request, &TTEST_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(4), Some(2)])
    );
    for dimension in [
        1025.0,
        f64::from_bits(2f64.powi(usize::BITS as i32).to_bits() - 1).floor(),
    ] {
        request.literals.literal_args[2] = LiteralValue::Number(dimension);
        let result = inference::infer(&request, &TTEST_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[2].shape, ShapeFact::Unknown);
    }
    request.literals = LiteralContext::default();
    assert_eq!(
        inference::infer(&request, &TTEST_CATALOG_ENTRY).outputs[1].shape,
        ShapeFact::Unknown
    );
}
