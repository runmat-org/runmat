use super::*;
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilityRequirement, CapabilitySet,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &STATSET_CATALOG_ENTRY,
    )
}

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

fn fields(output: &CallInference) -> &BTreeMap<String, ValueFact> {
    let ValueKindFact::Struct(structure) = &output.outputs[0].kind else {
        panic!("expected canonical options structure")
    };
    assert!(structure.fields_complete);
    assert_eq!(structure.fields.len(), StatOptionField::ALL.len());
    assert_eq!(structure.elements.first(), Some(&structure.fields));
    &structure.fields
}

#[test]
fn empty_call_has_twenty_known_empty_fields() {
    let result = call(vec![], vec![], 1);
    assert_eq!(fields(&result)["TolX"].shape.element_count(), Some(0));
}

#[test]
fn unknown_emptiness_does_not_invent_normalized_scalar_class() {
    let mut input = numeric(NumericClass::Single);
    input.shape = ShapeFact::from(vec![Some(1), None]);
    let result = call(
        vec![ValueFact::scalar(ValueKindFact::String), input],
        vec![LiteralValue::String("TolX".into()), LiteralValue::Unknown],
        1,
    );
    assert_eq!(fields(&result)["TolX"].kind, ValueKindFact::Unknown);
}

#[test]
fn singleton_integer_derivstep_does_not_guess_scalar_or_tensor_carrier() {
    let result = call(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::UInt64),
        ],
        vec![
            LiteralValue::String("DerivStep".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert_eq!(fields(&result)["DerivStep"].kind, ValueKindFact::Unknown);
    assert_eq!(fields(&result)["DerivStep"].shape.element_count(), Some(1));
}

#[test]
fn character_option_values_are_text_rows_not_numeric_scalars() {
    for field in ["Display", "UseParallel", "RobustWgtFun"] {
        for (shape, valid) in [
            (vec![Some(1), Some(2)], true),
            (vec![Some(2), Some(1)], false),
        ] {
            let value = ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(shape),
                StorageFact::Dense,
            );
            let result = call(
                vec![ValueFact::scalar(ValueKindFact::String), value],
                vec![LiteralValue::String(field.into()), LiteralValue::Unknown],
                1,
            );
            assert_eq!(result.diagnostics.is_empty(), valid);
        }
    }
}

#[test]
fn known_complex_tolerance_is_rejected_statically() {
    for field in ["TolX", "UseParallel"] {
        let value = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }));
        let result = call(
            vec![ValueFact::scalar(ValueKindFact::String), value],
            vec![LiteralValue::String(field.into()), LiteralValue::Unknown],
            1,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "RM.STATSET.INVALID_OPTION" }));
    }
}

#[test]
fn known_option_roles_normalize_only_their_selected_field() {
    let result = call(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::UInt64),
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::UInt8),
        ],
        vec![
            LiteralValue::String("MaxI".into()),
            LiteralValue::Unknown,
            LiteralValue::String("UseParallel".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert_eq!(
        fields(&result)["MaxIter"].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(fields(&result)["UseParallel"].kind, ValueKindFact::Logical);
    assert_eq!(fields(&result)["TolX"].shape.element_count(), Some(0));
}

#[test]
fn typed_derivstep_and_stored_callback_preserve_facts_without_invocation() {
    let mut callback_capabilities = CapabilitySet::default();
    callback_capabilities
        .0
        .insert(CapabilityRequirement::Network);
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId("unused".into()))),
        capabilities: callback_capabilities,
        parameters: vec![],
        parameters_complete: false,
        outputs: vec![],
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: vec![],
        captures_complete: true,
    }));
    let step = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = call(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            step.clone(),
            ValueFact::scalar(ValueKindFact::String),
            callback.clone(),
        ],
        vec![
            LiteralValue::String("DerivStep".into()),
            LiteralValue::Unknown,
            LiteralValue::String("OutputFcn".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert_eq!(
        fields(&result)["DerivStep"].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(fields(&result)["DerivStep"].shape, step.shape);
    assert_eq!(fields(&result)["OutputFcn"].kind, callback.kind);
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
}

#[test]
fn unknown_names_and_incomplete_source_do_not_invent_field_values() {
    let dynamic_name = call(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            numeric(NumericClass::Double),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        1,
    );
    assert_eq!(fields(&dynamic_name)["TolX"].kind, ValueKindFact::Unknown);
    let incomplete = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        false,
    )));
    let copied = call(vec![incomplete], vec![LiteralValue::Unknown], 1);
    assert_eq!(fields(&copied)["MaxIter"].kind, ValueKindFact::Unknown);
}

#[test]
fn bad_pairs_and_output_arity_are_diagnosed() {
    let bad = call(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("MaxIter".into())],
        2,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.STATSET.INVALID_ARGUMENT"));
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn known_invalid_option_class_and_non_scalar_count_are_diagnosed() {
    let wrong_class = call(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::String("MaxIter".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(wrong_class
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.STATSET.INVALID_OPTION"));
    let array = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let wrong_shape = call(
        vec![ValueFact::scalar(ValueKindFact::String), array],
        vec![
            LiteralValue::String("MaxIter".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(wrong_shape
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.STATSET.INVALID_OPTION"));
}
