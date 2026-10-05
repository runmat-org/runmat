use super::*;
use crate::*;
use runmat_types::{
    CallRequest, CellFact, DimensionFact, EffectKind, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn number(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        runmat_types::StorageFact::Dense,
    )
}

fn text(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::String,
        shape,
        runmat_types::StorageFact::Dense,
    )
}

fn shape(rows: usize, cols: usize) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(rows), DimensionFact::Known(cols)],
    }
}

#[test]
fn datetime_is_one_canonical_identity_with_clock_and_host_contract() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].identity.name, "datetime");
    assert_eq!(DATETIME_CATALOG_ENTRY.identity.name, "datetime");
    assert_eq!(
        DATETIME_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert!(DATETIME_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::Clock));
    assert_eq!(
        DATETIME_CATALOG_ENTRY.placement.residency,
        BuiltinResidencyPolicy::GatherToHost
    );
    assert_eq!(DATETIME_DESCRIPTOR.signatures.len(), 10);
    assert_eq!(DATETIME_CATALOG_ENTRY.documentation.examples.len(), 10);
    assert!(DATETIME_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .any(|example| example.id == "extract-components"
            && matches!(
                &example.verification,
                BuiltinExampleVerification::Assertions { .. }
            )));
    assert_eq!(DATETIME_EXTENSIONS.len(), 4);
    assert_eq!(DATETIME_INTEGER_CAPABILITIES.len(), 3);
}

#[test]
fn datetime_infers_date_vectors_text_and_component_shapes() {
    let vectors = inference::infer(
        &request(vec![number(shape(4, 6))], vec![], 1),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(vectors.outputs[0].shape, shape(4, 1));
    assert!(vectors.diagnostics.is_empty());
    let empty = inference::infer(
        &request(vec![number(shape(0, 3))], vec![], 1),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(empty.outputs[0].shape, shape(0, 1));
    let strings = inference::infer(
        &request(vec![text(shape(2, 1))], vec![], 1),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(strings.outputs[0].shape, shape(2, 1));
    let characters = inference::infer(
        &request(
            vec![ValueFact::proven(
                ValueKindFact::Character,
                shape(3, 10),
                runmat_types::StorageFact::Dense,
            )],
            vec![],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(characters.outputs[0].shape, shape(3, 1));
    let components = inference::infer(
        &request(
            vec![
                number(shape(2, 3)),
                number(ShapeFact::Scalar),
                number(shape(2, 3)),
            ],
            vec![],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(components.outputs[0].shape, shape(2, 3));
    assert!(components.diagnostics.is_empty());
    assert!(matches!(
        components.outputs[0].kind,
        ValueKindFact::Object(ObjectFact { .. })
    ));
    assert_eq!(
        components.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    );
}

#[test]
fn datetime_infers_cellstr_shape_and_rejects_known_nontext_elements() {
    let cell = |element: ValueFact| {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(element),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape(2, 1),
            runmat_types::StorageFact::Dense,
        )
    };
    let character = ValueFact::proven(
        ValueKindFact::Character,
        shape(1, 10),
        runmat_types::StorageFact::Dense,
    );
    let accepted = inference::infer(
        &request(
            vec![
                cell(character),
                text(ShapeFact::Scalar),
                text(ShapeFact::Scalar),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("InputFormat".into()),
                LiteralValue::String("yyyy-MM-dd".into()),
            ],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(accepted.outputs[0].shape, shape(2, 1));
    assert!(accepted.diagnostics.is_empty());

    let rejected = inference::infer(
        &request(vec![cell(number(ShapeFact::Scalar))], vec![], 1),
        &DATETIME_CATALOG_ENTRY,
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATETIME-INPUT"));

    for dims in [vec![0, 2], vec![2, 0], vec![2, 0, 3]] {
        let exact = ShapeFact::Shaped {
            dims: dims.iter().copied().map(DimensionFact::Known).collect(),
        };
        let input = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(
                    runmat_types::DynamicReason::RuntimeValue,
                )),
                elements: Vec::new(),
                elements_complete: true,
            }),
            exact.clone(),
            runmat_types::StorageFact::Dense,
        );
        let inferred = inference::infer(&request(vec![input], vec![], 1), &DATETIME_CATALOG_ENTRY);
        assert_eq!(inferred.outputs[0].shape, exact);
        assert!(inferred.diagnostics.is_empty());
    }
}

#[test]
fn datetime_option_forms_and_errors_are_explicit() {
    let conversion = inference::infer(
        &request(
            vec![
                number(shape(2, 1)),
                text(ShapeFact::Scalar),
                text(ShapeFact::Scalar),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("ConvertFrom".into()),
                LiteralValue::String("datenum".into()),
            ],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(conversion.outputs[0].shape, shape(2, 1));
    assert!(conversion.diagnostics.is_empty());
    let unsupported = inference::infer(
        &request(
            vec![
                number(shape(2, 1)),
                text(ShapeFact::Scalar),
                text(ShapeFact::Scalar),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("ConvertFrom".into()),
                LiteralValue::String("posixtime".into()),
            ],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert!(unsupported
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATETIME-CONVERSION"));
    let invalid_complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        runmat_types::StorageFact::Scalar,
    );
    assert!(!inference::infer(
        &request(vec![invalid_complex], vec![], 1),
        &DATETIME_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(!inference::infer(
        &request(
            vec![
                number(shape(2, 2)),
                number(shape(3, 2)),
                number(ShapeFact::Scalar)
            ],
            vec![],
            1
        ),
        &DATETIME_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(inference::infer(
        &request(
            vec![number(ShapeFact::Scalar), number(ShapeFact::Scalar)],
            vec![],
            1
        ),
        &DATETIME_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|diagnostic| diagnostic.code == "RM-DATETIME-ARITY"));

    let unknown_columns = inference::infer(
        &request(
            vec![number(ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(4), DimensionFact::Unknown],
            })],
            vec![],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(unknown_columns.outputs[0].shape, ShapeFact::Unknown);
    let input_format_on_numeric = inference::infer(
        &request(
            vec![
                number(shape(2, 1)),
                text(ShapeFact::Scalar),
                text(ShapeFact::Scalar),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("InputFormat".into()),
                LiteralValue::String("yyyy-MM-dd".into()),
            ],
            1,
        ),
        &DATETIME_CATALOG_ENTRY,
    );
    assert!(input_format_on_numeric
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATETIME-INPUT-FORMAT"));
}

#[test]
fn datetime_copy_and_discarded_outputs_retain_type() {
    let original = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DATETIME.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape(2, 4),
        runmat_types::StorageFact::Opaque,
    );
    let copy = inference::infer(
        &request(vec![original.clone()], vec![], 1),
        &DATETIME_CATALOG_ENTRY,
    );
    assert_eq!(copy.outputs[0].shape, original.shape);
    assert!(matches!(copy.outputs[0].kind, ValueKindFact::Object(_)));
    let discarded = inference::infer(&request(vec![original], vec![], 0), &DATETIME_CATALOG_ENTRY);
    assert!(discarded.outputs.is_empty());
}

#[test]
fn datetime_rejects_statically_known_other_object_classes() {
    for object in [
        ObjectFact {
            class: Some(runmat_types::ClassId(7)),
            runtime_class: None,
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: None,
        },
        ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::CALENDAR_DURATION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        },
    ] {
        let value = ValueFact::proven(
            ValueKindFact::Object(object),
            ShapeFact::Scalar,
            runmat_types::StorageFact::Opaque,
        );
        let inferred = inference::infer(&request(vec![value], vec![], 1), &DATETIME_CATALOG_ENTRY);
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DATETIME-INPUT"));
    }

    let unknown_class = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        runmat_types::StorageFact::Opaque,
    );
    let inferred = inference::infer(
        &request(vec![unknown_class], vec![], 1),
        &DATETIME_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
}
