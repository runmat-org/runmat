use super::*;
use runmat_types::*;

fn request(argument: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn raw_matrix_has_double_column_outputs() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(7), Some(4)]),
        StorageFact::Dense,
    );
    let inferred = crate::infer_catalog_call(&MACD_CATALOG_ENTRY, &request(data, 2));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs.len(), 2);
    for output in inferred.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(7), Some(1)]));
        assert_eq!(
            output.numeric().map(|numeric| numeric.class),
            Some(NumericClass::Double)
        );
    }
}

#[test]
fn known_table_kind_is_preserved_and_invalid_shape_is_diagnosed() {
    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TIMETABLE.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let inferred = crate::infer_catalog_call(&MACD_CATALOG_ENTRY, &request(table, 1));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref() == Some(&standard::TIMETABLE.owned()))
    );
    let bad = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(7), Some(3)]),
        StorageFact::Dense,
    );
    assert!(
        crate::infer_catalog_call(&MACD_CATALOG_ENTRY, &request(bad, 1))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-MACD-SHAPE")
    );
}

#[test]
fn former_undefined_example_is_executable_and_checked() {
    let example = &MACD_CATALOG_ENTRY.documentation.examples[0];
    assert!(example.program.contains("priceData ="));
    assert!(example.program.contains("fprintf"));
    assert!(example.display_output.is_some());
    assert!(matches!(
        example.verification,
        BuiltinExampleVerification::Assertions { .. }
    ));
}
