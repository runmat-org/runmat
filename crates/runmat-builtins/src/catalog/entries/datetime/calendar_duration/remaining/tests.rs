use super::*;
use runmat_types::*;
use std::collections::BTreeMap;
fn object(class: StaticClassIdentity, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    )
}
fn call(op: CalendarResidualOperation, args: Vec<ValueFact>, count: usize) -> CallInference {
    let request = CallRequest {
        arguments: args,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    };
    let entry = op.entry();
    match op {
        CalendarResidualOperation::IsDatetime => inference::predicate(&request, entry),
        CalendarResidualOperation::DateTick => inference::tick(&request, entry),
        CalendarResidualOperation::Plus => inference::plus(&request, entry),
        CalendarResidualOperation::Minus => inference::minus(&request, entry),
        _ => inference::comparison(&request, entry),
    }
}
#[test]
fn every_residual_identity_has_required_default_full_documentation_and_honest_frontier() {
    for op in [
        CalendarResidualOperation::IsDatetime,
        CalendarResidualOperation::DateTick,
        CalendarResidualOperation::Plus,
        CalendarResidualOperation::Minus,
        CalendarResidualOperation::Eq,
        CalendarResidualOperation::Ne,
    ] {
        let entry = op.entry();
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Incomplete);
        assert!(entry.contract.capabilities.is_empty());
        assert_eq!(
            entry.placement.portability,
            BuiltinPortability::NativeAndWasm
        );
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.documentation.examples.len(), 1);
        assert!(matches!(
            entry.documentation.examples[0].verification,
            BuiltinExampleVerification::Assertions { .. }
        ));
        let args = match op {
            CalendarResidualOperation::IsDatetime => {
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)]
            }
            CalendarResidualOperation::DateTick => vec![],
            _ => vec![object(standard::CALENDAR_DURATION, ShapeFact::Scalar); 2],
        };
        for count in [0, 1, 2] {
            let result = call(op, args.clone(), count);
            assert_eq!(result.diagnostics.is_empty(), count <= 1);
            assert_eq!(result.effects, entry.contract.effect_set());
            if count == 1 {
                assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
                assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
            }
        }
    }
}
#[test]
fn typed_calendar_facts_distinguish_datetime_addition_double_comparison_and_wrong_classes() {
    use CalendarResidualOperation as Op;
    let row = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)],
    };
    let lhs = object(standard::CALENDAR_DURATION, ShapeFact::Scalar);
    let rhs = object(standard::DATETIME, row.clone());
    let plus = call(Op::Plus, vec![lhs.clone(), rhs.clone()], 1);
    assert!(plus.diagnostics.is_empty());
    assert_eq!(plus.outputs[0].shape, row);
    assert!(
        matches!(&plus.outputs[0].kind,ValueKindFact::Object(o) if o.runtime_class.as_ref().is_some_and(|c|c.is(standard::DATETIME)))
    );
    for op in [Op::Minus, Op::Eq, Op::Ne] {
        assert!(!call(op, vec![lhs.clone(), rhs.clone()], 1)
            .diagnostics
            .is_empty());
    }
    for op in [Op::Eq, Op::Ne] {
        let result = call(op, vec![lhs.clone(), lhs.clone()], 1);
        assert!(
            matches!(&result.outputs[0].kind,ValueKindFact::Numeric(n) if n.class==NumericClass::Double)
        );
    }
    let col = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(3), DimensionFact::Known(1)],
    };
    assert!(!call(
        Op::Plus,
        vec![
            object(standard::CALENDAR_DURATION, row),
            object(standard::CALENDAR_DURATION, col)
        ],
        1
    )
    .diagnostics
    .is_empty());
    let unknown = call(
        Op::Plus,
        vec![lhs, ValueFact::unknown(DynamicReason::RuntimeValue)],
        1,
    );
    assert!(matches!(unknown.outputs[0].kind, ValueKindFact::Unknown));
    assert!(matches!(
        call(Op::IsDatetime, vec![rhs], 1).outputs[0].kind,
        ValueKindFact::Logical
    ));
}
