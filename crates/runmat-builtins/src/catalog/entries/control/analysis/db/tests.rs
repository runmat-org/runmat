use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, dims: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &DB_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn unary_and_text_modes_produce_real_host_output_with_signal_shape() {
    let signal = numeric(
        NumericClass::Single,
        NumericDomain::Complex,
        vec![Some(2), Some(3)],
    );
    for (args, literals) in [
        (vec![signal.clone()], vec![]),
        (
            vec![signal.clone(), ValueFact::scalar(ValueKindFact::String)],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("voltage".into()),
            ],
        ),
    ] {
        let result = infer(args, literals, 1);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, signal.shape);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    }
}

#[test]
fn resistance_broadcasts_and_native_single_selects_single_output() {
    let y = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        vec![Some(2), Some(1)],
    );
    let r = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        vec![Some(1), Some(3)],
    );
    let result = infer(vec![y, r], vec![], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real
        })
    );
}

#[test]
fn invalid_power_resistance_shape_and_mode_are_diagnosed() {
    let complex = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        vec![Some(2), Some(1)],
    );
    let text = ValueFact::scalar(ValueKindFact::String);
    let power = infer(
        vec![complex.clone(), text.clone()],
        vec![LiteralValue::Unknown, LiteralValue::String("power".into())],
        1,
    );
    assert!(power
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DB-POWER"));
    let uppercase_power = infer(
        vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(1), Some(1)],
            ),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![LiteralValue::Unknown, LiteralValue::String("POWER".into())],
        1,
    );
    assert!(
        uppercase_power.diagnostics.is_empty(),
        "{:?}",
        uppercase_power.diagnostics
    );
    let invalid_mode = infer(
        vec![complex.clone(), text],
        vec![LiteralValue::Unknown, LiteralValue::String("bad".into())],
        1,
    );
    assert!(invalid_mode
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DB-MODE"));
    let matrix_text = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert!(infer(
        vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(1), Some(1)]
            ),
            matrix_text
        ],
        vec![],
        1,
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-DB-MODE"));
    let invalid_r = infer(vec![complex.clone(), complex], vec![], 1);
    assert!(invalid_r
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DB-RESISTANCE"));
    let incompatible = infer(
        vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(2), Some(3)],
            ),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(4), Some(1)],
            ),
        ],
        vec![],
        1,
    );
    assert!(incompatible
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DB-SIZE"));
}

#[test]
fn dynamic_mode_stays_dynamic_and_arity_output_count_are_checked() {
    let y = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        vec![Some(2), Some(3)],
    );
    let result = infer(
        vec![y.clone(), ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![],
        1,
    );
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(infer(vec![], vec![], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DB-ARITY"));
    assert!(infer(vec![y.clone(), y.clone(), y.clone()], vec![], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DB-ARITY"));
    assert!(infer(vec![y.clone()], vec![], 0).outputs.is_empty());
    assert!(!infer(vec![y], vec![], 2).diagnostics.is_empty());
}
