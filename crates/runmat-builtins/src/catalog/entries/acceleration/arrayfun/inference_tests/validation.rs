use super::*;
use runmat_types::{InferenceSeverity, LiteralValue};

#[test]
fn rejects_non_callable_first_argument() {
    let result = infer(
        vec![
            numeric(NumericClass::Double, vec![Some(1), Some(1)]),
            numeric(NumericClass::Double, vec![Some(2), Some(2)]),
        ],
        LiteralContext::default(),
    );
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-CATALOG-ARRAYFUN-FUNCTION"
            && diagnostic.severity == InferenceSeverity::Error
    }));
}

#[test]
fn reports_incompatible_shapes() {
    let result = infer(
        vec![
            builtin("plus"),
            numeric(NumericClass::Double, vec![Some(2), Some(3)]),
            numeric(NumericClass::Double, vec![Some(4), Some(1)]),
        ],
        LiteralContext::default(),
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ARRAYFUN-SIZE"));
}

#[test]
fn dynamic_uniform_output_keeps_result_dynamic() {
    let literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("UniformOutput".into()),
        LiteralValue::Unknown,
    ]);
    let result = infer(
        vec![
            builtin("sin"),
            numeric(NumericClass::Single, vec![Some(1), Some(3)]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        literals,
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(result.outputs[0].shape.element_count(), Some(3));
}

#[test]
fn reports_unknown_options_and_invalid_controls() {
    let input = numeric(NumericClass::Double, vec![Some(1), Some(3)]);
    let unknown_literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("MysteryFlag".into()),
        LiteralValue::Bool(true),
    ]);
    let unknown = infer(
        vec![
            builtin("sin"),
            input.clone(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        unknown_literals,
    );
    assert!(unknown
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ARRAYFUN-OPTION"));

    let invalid = infer(
        vec![
            builtin("sin"),
            input,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::Number(2.0),
        ]),
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ARRAYFUN-UNIFORM-OUTPUT"));
}

#[test]
fn callback_capabilities_are_propagated() {
    let mut capability = CapabilitySet::default();
    capability
        .0
        .insert(runmat_types::CapabilityRequirement::Filesystem);
    let mut callback = callable(
        CallableIdentity::DynamicName(runmat_types::SymbolName("callback".into())),
        Some(ValueFact::scalar(ValueKindFact::Logical)),
    );
    let ValueKindFact::Callable(fact) = &mut callback.kind else {
        unreachable!();
    };
    fact.capabilities = capability;
    let result = infer(
        vec![
            callback,
            numeric(NumericClass::Double, vec![Some(1), Some(3)]),
        ],
        LiteralContext::default(),
    );
    assert!(result
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Filesystem));
    assert!(!result
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Accelerator));
}
