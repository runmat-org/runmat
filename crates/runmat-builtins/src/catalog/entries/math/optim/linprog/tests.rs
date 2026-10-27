use super::*;
use crate::OptimNumericInput;
use runmat_types::*;

#[test]
fn positional_facts_and_bounded_outputs_preserve_failure_geometry() {
    let real = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let empty = ValueFact::proven(
        real.kind.clone(),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![real, empty.clone(), empty],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    };
    let result = inference::infer(&request, &LINPROG_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(result.outputs[1].storage, StorageFact::Unknown);
    assert!(matches!(result.outputs[3].kind, ValueKindFact::Struct(_)));
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(usize::MAX));
    let invalid = inference::infer(&request, &LINPROG_CATALOG_ENTRY);
    assert!(invalid.outputs.is_empty());
    assert!(!invalid.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &LINPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn sparse_constraint_role_and_fifth_output_are_typed_and_failure_honest() {
    let real = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let f = ValueFact::proven(
        real.clone(),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let a = ValueFact::proven(
        real.clone(),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Sparse,
    );
    assert!(OptimNumericInput::OptionalSparseMatrix.accepts_metadata(&a));
    assert!(!OptimNumericInput::OptionalMatrix.accepts_metadata(&a));
    assert!(!OptimNumericInput::OptionalVector.accepts_metadata(&a));
    let request = CallRequest {
        arguments: vec![f, a, ValueFact::scalar(real)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    let result = inference::infer(&request, &LINPROG_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 5);
    let ValueKindFact::Struct(lambda) = &result.outputs[4].kind else {
        panic!("lambda struct")
    };
    assert!(lambda.fields_complete);
    assert_eq!(lambda.fields.len(), 4);
    for name in ["ineqlin", "eqlin", "lower", "upper"] {
        assert_eq!(lambda.fields[name].shape, ShapeFact::Unknown);
        assert_eq!(lambda.fields[name].storage, StorageFact::Dense);
    }
    let ValueKindFact::Struct(output) = &result.outputs[3].kind else {
        panic!("output struct")
    };
    for name in [
        "primalfeasibility",
        "dualfeasibility",
        "dualitygap",
        "firstorderopt",
    ] {
        assert!(output.fields.contains_key(name));
    }
}

#[test]
fn console_effects_follow_provable_default_options_not_dynamic_display_strings() {
    let real = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let empty = ValueFact::proven(
        real.kind.clone(),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![real, empty.clone(), empty.clone()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let has_console = |result: CallInference| {
        [
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
        ]
        .iter()
        .all(|effect| result.effects.0.contains(effect))
    };
    assert!(!has_console(inference::infer(
        &request,
        &LINPROG_CATALOG_ENTRY
    )));
    request.arguments.resize(8, empty);
    assert!(!has_console(inference::infer(
        &request,
        &LINPROG_CATALOG_ENTRY
    )));
    let mut fields = std::collections::BTreeMap::new();
    fields.insert(
        "MaxIter".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    request.arguments[7] = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        fields.clone(),
        true,
    )));
    assert!(!has_console(inference::infer(
        &request,
        &LINPROG_CATALOG_ENTRY
    )));
    fields.insert("DISPLAY".into(), ValueFact::scalar(ValueKindFact::String));
    request.arguments[7] =
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    assert!(has_console(inference::infer(
        &request,
        &LINPROG_CATALOG_ENTRY
    )));
    request.arguments[7] = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(has_console(inference::infer(
        &request,
        &LINPROG_CATALOG_ENTRY
    )));
}
