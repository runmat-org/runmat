use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

fn request(input: ValueFact, count: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(count),
    }
}

#[test]
fn known_nonscalar_paths_are_invalid_but_unknown_shape_is_not() {
    for fact in [
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(3), Some(1)]),
            StorageFact::Dense,
        ),
    ] {
        let invalid = inference::infer(
            &request(fact, RequestedOutputCount::One),
            &GEOMETRY_LOAD_CATALOG_ENTRY,
        );
        assert!(invalid
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GEOMETRY-LOAD-PATH"));
    }
    let unknown_rows = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![None, Some(3)]),
        StorageFact::Dense,
    );
    let result = inference::infer(
        &request(unknown_rows, RequestedOutputCount::One),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}

#[test]
fn load_preserves_typed_asset_and_rejects_numeric_path() {
    let valid = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            RequestedOutputCount::One,
        ),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let ValueKindFact::Object(object) = &valid.outputs[0].kind else {
        panic!("expected asset object")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&GEOMETRY_ASSET_CLASS.owned())
    );
    let unknown = inference::infer(
        &request(
            ValueFact::unknown(DynamicReason::RuntimeValue),
            RequestedOutputCount::One,
        ),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
    let invalid = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::Logical),
            RequestedOutputCount::Exactly(2),
        ),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GEOMETRY-LOAD-PATH"));
    assert!(!invalid.diagnostics.is_empty());
}

fn load_request(
    arguments: Vec<ValueFact>,
    literals: Vec<runmat_types::LiteralValue>,
) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn load_codes(arguments: Vec<ValueFact>, literals: Vec<runmat_types::LiteralValue>) -> Vec<String> {
    inference::infer(
        &load_request(arguments, literals),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    )
    .diagnostics
    .into_iter()
    .map(|diagnostic| diagnostic.code)
    .collect()
}

#[test]
fn units_option_accepts_supported_spellings() {
    use runmat_types::LiteralValue;
    let text = || ValueFact::scalar(ValueKindFact::String);
    for unit in ["m", "MM", "in", "Millimetres", "inches"] {
        let found = load_codes(
            vec![text(), text(), text()],
            vec![
                LiteralValue::String("part.brep".into()),
                LiteralValue::String("Units".into()),
                LiteralValue::String(unit.into()),
            ],
        );
        assert!(found.is_empty(), "{unit}: {found:?}");
    }
    // Dynamic option values cannot be checked statically and must not error.
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(load_codes(vec![text(), unknown.clone(), unknown], vec![]).is_empty());
}

#[test]
fn units_option_rejects_bad_pairs_names_values_and_repeats() {
    use runmat_types::LiteralValue;
    let text = || ValueFact::scalar(ValueKindFact::String);
    assert!(load_codes(vec![text(), text()], vec![])
        .contains(&"RM-CATALOG-GEOMETRY-LOAD-ARITY".to_string()));
    assert!(load_codes(
        vec![text(), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Scale".into()),
            LiteralValue::String("mm".into()),
        ],
    )
    .contains(&"RM-CATALOG-GEOMETRY-LOAD-OPTION".to_string()));
    assert!(load_codes(
        vec![text(), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Units".into()),
            LiteralValue::String("ft".into()),
        ],
    )
    .contains(&"RM-CATALOG-GEOMETRY-LOAD-UNITS".to_string()));
    assert!(load_codes(
        vec![text(), text(), ValueFact::scalar(ValueKindFact::Logical)],
        vec![],
    )
    .contains(&"RM-CATALOG-GEOMETRY-LOAD-UNITS".to_string()));
    assert!(load_codes(
        vec![text(), text(), text(), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Units".into()),
            LiteralValue::String("mm".into()),
            LiteralValue::String("units".into()),
            LiteralValue::String("m".into()),
        ],
    )
    .contains(&"RM-CATALOG-GEOMETRY-LOAD-OPTION".to_string()));
}
