use super::*;
use runmat_types::*;

fn request(literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: literals
            .iter()
            .map(|value| match value {
                LiteralValue::String(_) => ValueFact::scalar(ValueKindFact::String),
                _ => ValueFact::unknown(DynamicReason::RuntimeValue),
            })
            .collect(),
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn file_schema_is_a_host_table_not_a_guessed_scalar_struct() {
    for literal in [
        LiteralValue::String("measurements.parquet".into()),
        LiteralValue::Unknown,
    ] {
        let result = inference::infer(&request(vec![literal]), &PARQUETREAD_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert!(
            matches!(&result.outputs[0].kind, ValueKindFact::Object(fact)
            if fact.runtime_class == Some(standard::TABLE.owned()) && !fact.properties_complete)
        );
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
    let text = |s: &str| LiteralValue::String(s.into());
    let valid = request(vec![
        text("f.parquet"),
        text("OutputType"),
        text("timetable"),
        text("OutputType"),
        text("table"),
    ]);
    assert!(inference::infer(&valid, &PARQUETREAD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let mut paged_text = valid.clone();
    for index in [0, 1, 3] {
        let length = paged_text
            .literals
            .literal_string_at(index)
            .unwrap()
            .chars()
            .count();
        paged_text.arguments[index] = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(1), Some(length)]),
            StorageFact::Dense,
        );
    }
    assert!(inference::infer(&paged_text, &PARQUETREAD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    paged_text.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(1), Some(5)]);
    assert!(!inference::infer(&paged_text, &PARQUETREAD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    for literals in [
        vec![],
        vec![text("f"), text("RowGroups")],
        vec![text("f"), text("Unknown"), LiteralValue::Unknown],
        vec![text("f"), text("OutputType"), text("timetable")],
        vec![text("f"), text("RowFilter"), LiteralValue::Unknown],
    ] {
        assert!(
            !inference::infer(&request(literals), &PARQUETREAD_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
    assert_eq!(
        ParquetReadControl::parse("rOwGrOuPs"),
        Some(ParquetReadControl::RowGroups)
    );
    assert_eq!(
        PARQUETREAD_CATALOG_ENTRY.placement.portability,
        crate::BuiltinPortability::NativeOnly
    );
}
