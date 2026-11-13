use super::*;
use crate::REQUIRED_DEFAULT_BINDING;
use runmat_types::{
    CallRequest, DimensionFact, DynamicReason, LiteralContext, NumericClass, NumericDomain,
    NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![]),
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

fn duration(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::CALENDAR_DURATION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        runmat_types::StorageFact::Opaque,
    )
}

#[test]
fn extracted_declarations_keep_all_public_identity_and_extension_exports() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.identity.name)
            .collect::<Vec<_>>(),
        [
            "caldays",
            "calendarDuration",
            "calmonths",
            "calquarters",
            "calweeks",
            "calyears",
            "iscalendarduration",
            "isdatetime",
            "datetick",
            "calendarDuration.plus",
            "calendarDuration.minus",
            "calendarDuration.eq",
            "calendarDuration.ne",
        ]
    );
    assert_eq!(CALENDAR_DURATION_EXTENSIONS.len(), 3);
    assert_eq!(
        CALENDAR_UNIT_EXTENSIONS,
        [CALENDAR_DURATION_GPU_INPUT_EXTENSION]
    );
    assert_eq!(
        CALENDARDURATION_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(
        ISCALENDARDURATION_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
}

#[test]
fn unit_forms_infer_constructor_object_or_double_extraction() {
    let shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)],
    };
    for entry in [
        &CALDAYS_CATALOG_ENTRY,
        &CALWEEKS_CATALOG_ENTRY,
        &CALMONTHS_CATALOG_ENTRY,
        &CALQUARTERS_CATALOG_ENTRY,
        &CALYEARS_CATALOG_ENTRY,
    ] {
        let built = inference::unit(&call(vec![number(shape.clone())], 1), entry);
        assert!(
            built.diagnostics.is_empty(),
            "{}: {:?}",
            entry.identity.name,
            built.diagnostics
        );
        assert_eq!(built.outputs[0].shape, shape);
        assert!(matches!(built.outputs[0].kind, ValueKindFact::Object(_)));
        let extracted = inference::unit(&call(vec![duration(shape.clone())], 1), entry);
        assert_eq!(extracted.outputs[0].shape, shape);
        assert!(matches!(
            extracted.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
        assert!(!inference::unit(&call(vec![], 1), entry)
            .diagnostics
            .is_empty());
        assert!(!inference::unit(
            &call(vec![ValueFact::scalar(ValueKindFact::String)], 1),
            entry
        )
        .diagnostics
        .is_empty());
        assert!(inference::unit(
            &call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
            entry
        )
        .diagnostics
        .is_empty());
        assert!(
            inference::unit(&call(vec![number(shape.clone())], 0), entry)
                .outputs
                .is_empty()
        );
    }
}

#[test]
fn constructor_and_predicate_keep_shape_and_class_facts() {
    let component_arities: Vec<_> = contracts::CALENDARDURATION
        .signatures
        .iter()
        .filter(|signature| signature.label.contains("Y,M,D"))
        .map(|signature| signature.inputs.len())
        .collect();
    assert_eq!(component_arities, vec![3, 6]);
    let matrix = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(4), DimensionFact::Known(3)],
    };
    let inferred = inference::constructor(
        &call(vec![number(matrix)], 1),
        &CALENDARDURATION_CATALOG_ENTRY,
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(4), DimensionFact::Known(1)],
        }
    );
    let empty_matrix = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(0), DimensionFact::Known(3)],
    };
    let empty = inference::constructor(
        &call(vec![number(empty_matrix.clone())], 1),
        &CALENDARDURATION_CATALOG_ENTRY,
    );
    assert_eq!(
        empty.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(0), DimensionFact::Known(1)],
        }
    );
    let empty_unit = inference::unit(
        &call(vec![number(empty_matrix.clone())], 1),
        &CALDAYS_CATALOG_ENTRY,
    );
    assert_eq!(empty_unit.outputs[0].shape, empty_matrix);
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Object(_)));
    let predicate = inference::predicate(
        &call(vec![duration(ShapeFact::Scalar)], 1),
        &ISCALENDARDURATION_CATALOG_ENTRY,
    );
    assert_eq!(predicate.outputs[0].shape, ShapeFact::Scalar);
    assert!(matches!(predicate.outputs[0].kind, ValueKindFact::Logical));
    assert!(!inference::constructor(
        &call(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &CALENDARDURATION_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(
        !inference::predicate(&call(vec![], 1), &ISCALENDARDURATION_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
