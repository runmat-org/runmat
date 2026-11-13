use super::*;
use crate::{infer_catalog_call, BuiltinCatalogEntry, BuiltinResidencyPolicy};
use runmat_types::*;

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::default(),
    }
}
fn property<'a>(output: &'a ValueFact, name: &str) -> &'a ValueFact {
    let ValueKindFact::Object(object) = &output.kind else {
        panic!("expected descriptor object");
    };
    object.properties.get(name).expect("declared property")
}
fn check(
    entry: &BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> CallInference {
    infer_catalog_call(entry, &request(arguments, literals))
}
fn text_literal(value: &str) -> LiteralValue {
    LiteralValue::String(value.into())
}

#[test]
fn six_identities_have_host_scalar_classes_without_invented_services() {
    for (entry, count) in [
        (&TIMERANGE_ENTRY, 2),
        (&VARTYPE_ENTRY, 1),
        (&ROWFILTER_ENTRY, 0),
        (&ARRAY_DATASTORE_ENTRY, 1),
        (&PARQUET_DATASTORE_ENTRY, 1),
        (&UITABLE_ENTRY, 0),
    ] {
        let args = if entry.identity.name == "vartype" {
            vec![ValueFact::scalar(ValueKindFact::String)]
        } else {
            vec![unknown(); count]
        };
        let result = check(entry, args, Vec::new());
        assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(result.outputs[0].storage, StorageFact::Opaque);
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("object");
        };
        assert_eq!(
            object.runtime_class,
            Some(StaticClassIdentity::new(entry.identity.name).owned())
        );
        assert!(object.properties_complete);
        assert_eq!(object.handle_semantics, Some(false));
        assert!(entry.contract.capabilities.is_empty());
        assert_eq!(
            entry.placement.residency,
            BuiltinResidencyPolicy::GatherToHost
        );
        assert!(!entry.documentation.examples.is_empty());
    }
}

#[test]
fn fixed_output_selection_keeps_validation_and_rejects_extra_outputs() {
    for (entry, count) in [
        (&TIMERANGE_ENTRY, 2),
        (&VARTYPE_ENTRY, 1),
        (&ROWFILTER_ENTRY, 0),
        (&ARRAY_DATASTORE_ENTRY, 1),
        (&PARQUET_DATASTORE_ENTRY, 1),
        (&UITABLE_ENTRY, 0),
    ] {
        let mut call = request(vec![unknown(); count], Vec::new());
        call.outputs = OutputSelection::new(RequestedOutputCount::Zero);
        let discarded = infer_catalog_call(entry, &call);
        assert!(discarded.outputs.is_empty(), "{}", entry.identity.name);
        call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer_catalog_call(entry, &call).diagnostics.is_empty());
    }
    let mut invalid = request(Vec::new(), Vec::new());
    invalid.outputs = OutputSelection::new(RequestedOutputCount::Zero);
    assert!(!infer_catalog_call(&ARRAY_DATASTORE_ENTRY, &invalid)
        .diagnostics
        .is_empty());
}

#[test]
fn timerange_preserves_exact_bound_classes_and_checks_interval_not_bound_contents() {
    let bound = numeric(NumericClass::UInt64, ShapeFact::Scalar);
    let result = check(
        &TIMERANGE_ENTRY,
        vec![bound.clone(), bound.clone()],
        vec![
            LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Integer {
                text: "18446744073709551615".into(),
                class: NumericClass::UInt64,
            },
        ],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(property(&result.outputs[0], "Start").kind, bound.kind);
    assert_eq!(property(&result.outputs[0], "End").shape, ShapeFact::Scalar);
    for literal in ["openright", "OPENLEFT", "open", "closed"] {
        assert!(check(
            &TIMERANGE_ENTRY,
            vec![
                unknown(),
                unknown(),
                ValueFact::scalar(ValueKindFact::String)
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                text_literal(literal)
            ]
        )
        .diagnostics
        .is_empty());
    }
    assert!(!check(&TIMERANGE_ENTRY, vec![unknown()], Vec::new())
        .diagnostics
        .is_empty());
    assert!(!check(
        &TIMERANGE_ENTRY,
        vec![
            unknown(),
            unknown(),
            ValueFact::scalar(ValueKindFact::String)
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            text_literal("bad")
        ]
    )
    .diagnostics
    .is_empty());
    assert!(!check(
        &TIMERANGE_ENTRY,
        vec![
            unknown(),
            unknown(),
            numeric(NumericClass::Double, ShapeFact::Scalar)
        ],
        Vec::new()
    )
    .diagnostics
    .is_empty());
}

#[test]
fn vartype_accepts_character_rows_but_not_numeric_or_text_arrays() {
    let row = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(7)]),
        StorageFact::Dense,
    );
    assert!(check(&VARTYPE_ENTRY, vec![row], Vec::new())
        .diagnostics
        .is_empty());
    for value in [
        numeric(NumericClass::UInt64, ShapeFact::Scalar),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(7)]),
            StorageFact::Dense,
        ),
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
    ] {
        assert!(!check(&VARTYPE_ENTRY, vec![value], Vec::new())
            .diagnostics
            .is_empty());
    }
    assert!(!check(&VARTYPE_ENTRY, Vec::new(), Vec::new())
        .diagnostics
        .is_empty());
}

#[test]
fn rowfilter_has_defaults_retains_nested_integer_payloads_and_rejects_direct_controls() {
    let default = check(&ROWFILTER_ENTRY, Vec::new(), Vec::new());
    assert_eq!(
        property(&default.outputs[0], "Variables").shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    let nested = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(numeric(NumericClass::UInt64, ShapeFact::Scalar)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let result = check(
        &ROWFILTER_ENTRY,
        vec![nested.clone(), ValueFact::scalar(ValueKindFact::Logical)],
        Vec::new(),
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(property(&result.outputs[0], "Variables").kind, nested.kind);
    assert!(!check(
        &ROWFILTER_ENTRY,
        vec![numeric(NumericClass::Int32, ShapeFact::Scalar)],
        Vec::new()
    )
    .diagnostics
    .is_empty());
    assert!(!check(
        &ROWFILTER_ENTRY,
        vec![unknown(), unknown(), unknown()],
        Vec::new()
    )
    .diagnostics
    .is_empty());
}

#[test]
fn array_datastore_preserves_resident_integer_nd_data_and_checks_option_roles() {
    let mut data = numeric(
        NumericClass::UInt64,
        ShapeFact::from(vec![Some(0), Some(2), Some(3)]),
    );
    data.residency = ResidencyFact::Device { provider: None };
    let result = check(&ARRAY_DATASTORE_ENTRY, vec![data.clone()], Vec::new());
    let stored = property(&result.outputs[0], "Data");
    assert_eq!(stored.kind, data.kind);
    assert_eq!(stored.shape, data.shape);
    assert_eq!(stored.residency, ResidencyFact::Host);
    assert_eq!(
        property(&result.outputs[0], "ReadSize").kind,
        numeric(NumericClass::Double, ShapeFact::Scalar).kind
    );
    for class in [NumericClass::Int64, NumericClass::Single] {
        assert!(!check(
            &ARRAY_DATASTORE_ENTRY,
            vec![
                unknown(),
                ValueFact::scalar(ValueKindFact::String),
                numeric(class, ShapeFact::Scalar)
            ],
            vec![
                LiteralValue::Unknown,
                text_literal("ReadSize"),
                LiteralValue::Number(2.0)
            ]
        )
        .diagnostics
        .is_empty());
    }
    assert!(!check(
        &ARRAY_DATASTORE_ENTRY,
        vec![unknown(), ValueFact::scalar(ValueKindFact::String)],
        Vec::new()
    )
    .diagnostics
    .is_empty());
    assert!(!check(
        &ARRAY_DATASTORE_ENTRY,
        vec![
            unknown(),
            ValueFact::scalar(ValueKindFact::String),
            unknown()
        ],
        vec![
            LiteralValue::Unknown,
            text_literal("unknown"),
            LiteralValue::Unknown
        ]
    )
    .diagnostics
    .is_empty());
    let mismatch = check(
        &ARRAY_DATASTORE_ENTRY,
        vec![
            unknown(),
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::Double, ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            text_literal("IterationDimension"),
            LiteralValue::Number(2.0),
            text_literal("OutputType"),
            text_literal("same"),
        ],
    );
    assert!(!mismatch.diagnostics.is_empty());
    let dynamic = check(
        &ARRAY_DATASTORE_ENTRY,
        vec![unknown(), unknown(), unknown()],
        Vec::new(),
    );
    assert!(dynamic.diagnostics.is_empty());
}

#[test]
fn parquet_read_size_is_option_dependent_and_sparse_integer_payloads_reject() {
    let default = check(&PARQUET_DATASTORE_ENTRY, vec![unknown()], Vec::new());
    assert_eq!(
        property(&default.outputs[0], "ReadSize").kind,
        ValueKindFact::String
    );
    let numeric_size = check(
        &PARQUET_DATASTORE_ENTRY,
        vec![
            unknown(),
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::Double, ShapeFact::Scalar),
        ],
        vec![
            LiteralValue::Unknown,
            text_literal("ReadSize"),
            LiteralValue::Number(3.0),
        ],
    );
    assert!(numeric_size.diagnostics.is_empty());
    assert_eq!(
        property(&numeric_size.outputs[0], "ReadSize").kind,
        numeric(NumericClass::Double, ShapeFact::Scalar).kind
    );
    let dynamic = check(
        &PARQUET_DATASTORE_ENTRY,
        vec![unknown(), unknown(), unknown()],
        Vec::new(),
    );
    assert_eq!(
        property(&dynamic.outputs[0], "ReadSize").kind,
        ValueKindFact::Unknown
    );
    let mut integer = numeric(NumericClass::Int64, ShapeFact::from(vec![Some(2), Some(1)]));
    integer.storage = StorageFact::Sparse;
    let nested = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(integer),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    assert!(!check(&PARQUET_DATASTORE_ENTRY, vec![nested], Vec::new())
        .diagnostics
        .is_empty());
    assert!(!check(
        &PARQUET_DATASTORE_ENTRY,
        vec![
            unknown(),
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::Double, ShapeFact::Scalar)
        ],
        vec![
            LiteralValue::Unknown,
            text_literal("BlockSize"),
            LiteralValue::Number(0.0)
        ]
    )
    .diagnostics
    .is_empty());
}

#[test]
fn uitable_selects_first_data_option_and_does_not_invent_unknown_payloads() {
    let first = numeric(
        NumericClass::UInt64,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let last = numeric(NumericClass::Single, ShapeFact::Scalar);
    let result = check(
        &UITABLE_ENTRY,
        vec![
            ValueFact::scalar(ValueKindFact::String),
            first.clone(),
            ValueFact::scalar(ValueKindFact::String),
            last,
        ],
        vec![
            text_literal("Data"),
            LiteralValue::Unknown,
            text_literal("Data"),
            LiteralValue::Unknown,
        ],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(property(&result.outputs[0], "Data").kind, first.kind);
    let result = check(&UITABLE_ENTRY, vec![unknown(), first], Vec::new());
    assert_eq!(
        property(&result.outputs[0], "Data").kind,
        ValueKindFact::Unknown
    );
    assert_eq!(
        property(&result.outputs[0], "ColumnName").shape,
        ShapeFact::from(vec![Some(0), Some(1)])
    );
}
