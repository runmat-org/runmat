use super::*;
use runmat_types::{
    CallRequest, ClassId, DimensionFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
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
        StorageFact::Dense,
    )
}
fn logical(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
}
fn shape(rows: usize, cols: usize) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(rows), DimensionFact::Known(cols)],
    }
}

#[test]
fn duration_inventory_is_exact_and_all_examples_have_assertions() {
    assert_eq!(ENTRIES.len(), 18);
    for entry in ENTRIES {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(
            !entry.documentation.examples.is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(
            entry.documentation.examples.iter().all(|example| matches!(
                example.verification,
                BuiltinExampleVerification::Assertions { .. }
            )),
            "{}",
            entry.identity.name
        );
        assert_eq!(entry.bindings.len(), 1, "{}", entry.identity.name);
    }
}

#[test]
fn duration_diagnostics_match_each_public_operation_and_predicate_is_total() {
    use runmat_types::EffectKind;

    let expected = [
        (
            "duration",
            &[
                DurationErrorKind::InvalidArgument,
                DurationErrorKind::InvalidFormat,
                DurationErrorKind::ShapeMismatch,
                DurationErrorKind::Internal,
            ][..],
        ),
        (
            "days",
            &[
                DurationErrorKind::InvalidArgument,
                DurationErrorKind::InvalidObject,
                DurationErrorKind::OutOfRange,
                DurationErrorKind::Internal,
            ][..],
        ),
        (
            "duration.subsref",
            &[
                DurationErrorKind::InvalidArgument,
                DurationErrorKind::InvalidIndex,
                DurationErrorKind::InvalidObject,
                DurationErrorKind::Internal,
            ][..],
        ),
        (
            "duration.subsasgn",
            &[
                DurationErrorKind::InvalidArgument,
                DurationErrorKind::InvalidFormat,
                DurationErrorKind::InvalidIndex,
                DurationErrorKind::InvalidObject,
                DurationErrorKind::Internal,
            ][..],
        ),
        (
            "duration.plus",
            &[
                DurationErrorKind::InvalidArgument,
                DurationErrorKind::InvalidObject,
                DurationErrorKind::ShapeMismatch,
                DurationErrorKind::Internal,
            ][..],
        ),
    ];
    for (name, kinds) in expected {
        let entry = ENTRIES
            .iter()
            .find(|entry| entry.identity.name == name)
            .unwrap();
        let identifiers = entry
            .descriptor
            .errors
            .iter()
            .map(|error| error.identifier.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            identifiers,
            kinds
                .iter()
                .map(|kind| kind.identifier())
                .collect::<Vec<_>>(),
            "{name}"
        );
        assert!(
            entry.contract.effects.contains(&EffectKind::MayThrow),
            "{name}"
        );
    }
    for entry in ENTRIES {
        if entry.identity.name == "isduration" {
            assert!(entry.descriptor.errors.is_empty());
            assert!(!entry.contract.effects.contains(&EffectKind::MayThrow));
        } else {
            assert!(
                !entry.descriptor.errors.is_empty(),
                "{}",
                entry.identity.name
            );
            assert!(
                entry.contract.effects.contains(&EffectKind::MayThrow),
                "{}",
                entry.identity.name
            );
        }
    }
}

#[test]
fn assignment_descriptor_covers_transitive_duration_read_and_text_format_errors() {
    let errors = DURATION_SUBSASGN_DESCRIPTOR.errors;
    for kind in [
        DurationErrorKind::InvalidObject,
        DurationErrorKind::Internal,
    ] {
        assert!(errors
            .iter()
            .any(|error| error.identifier == Some(kind.identifier())));
    }
    let format = errors
        .iter()
        .find(|error| error.identifier == Some(DurationErrorKind::InvalidFormat.identifier()))
        .expect("Format diagnostic");
    assert_eq!(
        format.when,
        "A textual Format option or property names an unsupported display format."
    );
}

#[test]
fn constructor_documents_short_form_and_retains_visible_legacy_examples() {
    let short_form = DURATION_DESCRIPTOR
        .signatures
        .iter()
        .find(|signature| signature.label == "t = duration(hours)")
        .expect("one-component RunMat form");
    assert_eq!(short_form.inputs.len(), 1);
    assert_eq!(short_form.inputs[0].name, "hours");
    assert_eq!(short_form.outputs.len(), 1);
    assert!(DURATION_EXTENSIONS
        .iter()
        .any(|extension| extension.id == "duration-short-component-form"));

    let examples = DURATION_CATALOG_ENTRY.documentation.examples;
    for (id, visible_statement) in [
        ("components", "disp(t)"),
        ("broadcast", "disp(t)"),
        ("format", "disp(t)"),
        ("datetime", "disp(t1)"),
        ("units", "fprintf('h = %g"),
    ] {
        let example = examples
            .iter()
            .find(|example| example.id == id)
            .expect("preserved legacy example");
        assert!(example.program.contains(visible_statement), "{id}");
        assert!(example.display_output.is_some(), "{id}");
    }
}

#[test]
fn constructor_and_units_prove_shapes_without_guessing_unknown_class() {
    let built =
        crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &call(vec![number(shape(4, 3))], 1));
    assert!(built.diagnostics.is_empty());
    assert_eq!(built.outputs[0].shape, shape(4, 1));
    assert_eq!(facts::duration_class(&built.outputs[0]), Some(true));

    let unit = crate::infer_catalog_call(
        &HOURS_CATALOG_ENTRY,
        &call(vec![built.outputs[0].clone()], 1),
    );
    assert_eq!(unit.outputs[0].shape, shape(4, 1));
    assert!(matches!(
        unit.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));

    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = crate::infer_catalog_call(&DAYS_CATALOG_ENTRY, &call(vec![unknown], 1));
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
    assert!(!dynamic.dynamic_outputs);
}

#[test]
fn constructor_format_pair_does_not_corrupt_component_inference() {
    let mut request = call(
        vec![
            number(shape(1, 2)),
            number(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        1,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Format".into()),
        LiteralValue::String("hh:mm".into()),
    ]);
    let inferred = crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &request);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].shape, shape(1, 2));
    let two_outputs =
        crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &call(vec![number(shape(2, 3))], 2));
    assert!(!two_outputs.diagnostics.is_empty());
}

#[test]
fn logical_constructor_inputs_match_runtime_admission() {
    for inputs in [
        vec![logical(shape(2, 3))],
        vec![
            logical(shape(1, 2)),
            number(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
        ],
        vec![
            logical(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
        ],
    ] {
        let inferred = crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &call(inputs, 1));
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(facts::duration_class(&inferred.outputs[0]), Some(true));
    }
}

#[test]
fn constructor_known_shape_conflict_is_diagnosed_without_guessing_unknown_extents() {
    let conflict = crate::infer_catalog_call(
        &DURATION_CATALOG_ENTRY,
        &call(
            vec![
                number(shape(2, 1)),
                number(shape(3, 1)),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
    );
    assert!(conflict
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "RM-DURATION-COMPONENT-SHAPE" }));

    let uncertain = ShapeFact::Shaped {
        dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)],
    };
    let dynamic = crate::infer_catalog_call(
        &DURATION_CATALOG_ENTRY,
        &call(
            vec![
                number(shape(2, 1)),
                number(uncertain),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
    );
    assert!(!dynamic
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "RM-DURATION-COMPONENT-SHAPE" }));
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn known_other_object_class_is_not_inferred_as_duration() {
    let other = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: Some(ClassId(7)),
            runtime_class: None,
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    assert_eq!(facts::duration_class(&other), Some(false));
    assert!(
        !crate::infer_catalog_call(&SECONDS_CATALOG_ENTRY, &call(vec![other], 1))
            .diagnostics
            .is_empty()
    );
}

#[test]
fn comparison_result_is_logical_and_broadcasted() {
    let left = facts::duration(shape(2, 1));
    let right = facts::duration(shape(1, 3));
    let inferred =
        crate::infer_catalog_call(&DURATION_EQ_CATALOG_ENTRY, &call(vec![left, right], 1));
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].shape, shape(2, 3));
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Logical));
}

#[test]
fn comparison_method_rejects_two_known_numeric_operands() {
    let inferred = crate::infer_catalog_call(
        &DURATION_EQ_CATALOG_ENTRY,
        &call(vec![number(shape(1, 1)), number(shape(1, 1))], 1),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DURATION-OP-RECEIVER"));
}

#[test]
fn arithmetic_handlers_select_typed_operation_independently_of_identity_labels() {
    let lhs = facts::duration(shape(2, 1));
    let rhs = facts::datetime(shape(1, 3));
    // A diagnostic label is not authority to change the registered handler.
    let mut addition = DURATION_PLUS_CATALOG_ENTRY;
    addition.identity.name = "duration.minus";
    let mut subtraction = DURATION_MINUS_CATALOG_ENTRY;
    subtraction.identity.name = "duration.plus";
    let request = call(vec![lhs.clone(), rhs], 1);
    let added = crate::infer_catalog_call(&addition, &request);
    assert!(added.diagnostics.is_empty());
    assert_eq!(added.outputs[0].shape, shape(2, 3));
    assert!(
        matches!(&added.outputs[0].kind, ValueKindFact::Object(object)
        if object.runtime_class.as_ref().is_some_and(|class| class.is(runmat_types::standard::DATETIME)))
    );
    let rejected = crate::infer_catalog_call(&subtraction, &request);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|error| error.code == "RM-DURATION-OP-RIGHT"));
    for rhs in [
        number(shape(1, 3)),
        logical(shape(1, 3)),
        facts::duration(shape(1, 3)),
    ] {
        for entry in [&addition, &subtraction] {
            let result = crate::infer_catalog_call(entry, &call(vec![lhs.clone(), rhs.clone()], 1));
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].shape, shape(2, 3));
            assert_eq!(facts::duration_class(&result.outputs[0]), Some(true));
        }
    }
    for entry in [&addition, &subtraction] {
        let mut request = call(
            vec![lhs.clone(), ValueFact::unknown(DynamicReason::RuntimeValue)],
            1,
        );
        request.literals =
            LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
        let dynamic = crate::infer_catalog_call(entry, &request);
        assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
        assert!(crate::infer_catalog_call(entry, &request)
            .outputs
            .is_empty());
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!crate::infer_catalog_call(entry, &request)
            .diagnostics
            .is_empty());
    }
}
