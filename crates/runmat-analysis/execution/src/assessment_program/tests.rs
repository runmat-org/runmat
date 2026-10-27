use super::*;
use runmat_execution::value::StructField;
fn function_facts() -> FunctionAnalysis {
    use runmat_types::{
        CallableFact, CapabilitySet, EffectSet, NumericClass, NumericFact, ProgramFunctionId,
        StructFact,
    };
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        std::collections::BTreeMap::from([
            (
                "evaluated".into(),
                ValueFact::scalar(ValueKindFact::Logical),
            ),
            ("accepted".into(), ValueFact::scalar(ValueKindFact::Logical)),
        ]),
        true,
    )));
    FunctionAnalysis {
        function: ProgramFunctionId(1),
        captures: Vec::new(),
        callable: CallableFact {
            capabilities: CapabilitySet::default(),
            identity: None,
            parameters: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))],
            parameters_complete: true,
            outputs: vec![output.clone()],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: true,
        },
        outputs: vec![output],
        effects: EffectSet::default(),
        cooperative_suspension: false,
        capabilities: CapabilitySet::default(),
        convergence: FunctionConvergence::Exact,
    }
}
#[test]
fn explicit_singleton_shapes_use_the_shared_scalar_contract() {
    let mut facts = function_facts();
    facts.callable.parameters[0].shape = runmat_types::ShapeFact::from(vec![Some(1), Some(1)]);
    let mut output = facts.outputs[0].clone();
    output.shape = runmat_types::ShapeFact::from(vec![Some(1), Some(1)]);
    let ValueKindFact::Struct(record) = &mut output.kind else {
        unreachable!()
    };
    for field in record.fields.values_mut() {
        field.shape = runmat_types::ShapeFact::from(vec![Some(1), Some(1)]);
    }
    facts.outputs = vec![output.clone()];
    facts.callable.outputs = vec![output];
    assert!(validate_assessment_function(&facts, 1).is_ok());
    facts.callable.parameters[0].shape = runmat_types::ShapeFact::Ranked { rank: 2 };
    assert!(validate_assessment_function(&facts, 1).is_err());
}
#[test]
fn exact_contract_rejects_unknown_outputs_captures_and_widening() {
    let exact = function_facts();
    validate_assessment_function(&exact, 1).unwrap();
    assert!(validate_assessment_function(&exact, 2).is_err());
    let mut widened = exact.clone();
    widened.convergence = FunctionConvergence::Widened;
    assert!(validate_assessment_function(&widened, 1).is_err());
    let mut captured = exact.clone();
    captured
        .callable
        .captures
        .push(exact.callable.parameters[0].clone());
    assert!(validate_assessment_function(&captured, 1).is_err());
    let mut incomplete = exact.clone();
    let ValueKindFact::Struct(record) = &mut incomplete.outputs[0].kind else {
        unreachable!()
    };
    record.fields_complete = false;
    assert!(validate_assessment_function(&incomplete, 1).is_err());
    let mut impure = exact;
    impure.effects.0.insert(EffectKind::Randomness);
    assert!(validate_assessment_function(&impure, 1).is_err());
}
fn record(fields: &[(&str, InlineValue)]) -> ValuePayload {
    let mut fields = fields
        .iter()
        .map(|(name, value)| StructField {
            name: (*name).into(),
            value: ValuePayload::Inline(Box::new(value.clone())),
        })
        .collect::<Vec<_>>();
    fields.sort_by(|left, right| left.name.cmp(&right.name));
    ValuePayload::Inline(Box::new(InlineValue::Struct(fields)))
}
#[test]
fn outcome_distinguishes_failure_from_absent_evidence() {
    assert_eq!(
        decode_assessment_outcome(&record(&[
            ("evaluated", InlineValue::Logical(true)),
            ("accepted", InlineValue::Logical(false))
        ]))
        .unwrap(),
        AssessmentFunctionOutcome {
            evaluated: true,
            accepted: false
        }
    );
    assert_eq!(
        decode_assessment_outcome(&record(&[
            ("accepted", InlineValue::Logical(true)),
            ("evaluated", InlineValue::Logical(false))
        ]))
        .unwrap(),
        AssessmentFunctionOutcome {
            evaluated: false,
            accepted: true
        }
    );
}
#[test]
fn malformed_outcomes_cannot_become_criterion_results() {
    for fields in [
        vec![("accepted", InlineValue::Logical(true))],
        vec![
            ("evaluated", InlineValue::Logical(true)),
            ("accepted", InlineValue::F64Bits(1.0_f64.to_bits())),
        ],
        vec![
            ("accepted", InlineValue::Logical(true)),
            ("accepted", InlineValue::Logical(false)),
        ],
        vec![
            ("evaluated", InlineValue::Logical(true)),
            ("accepted", InlineValue::Logical(true)),
            ("extra", InlineValue::Logical(false)),
        ],
    ] {
        assert!(decode_assessment_outcome(&record(&fields)).is_err());
    }
}

#[test]
fn outcome_accepts_logical_singletons_without_numeric_coercion() {
    use runmat_execution::value::DenseValue;
    let dense = |element_type, shape, little_endian_data| {
        InlineValue::Dense(DenseValue {
            element_type,
            shape,
            little_endian_data,
        })
    };
    let decode = |accepted| {
        decode_assessment_outcome(&record(&[
            ("evaluated", InlineValue::Logical(true)),
            ("accepted", accepted),
        ]))
    };
    for (byte, accepted) in [(0, false), (1, true)] {
        assert_eq!(
            decode(dense(ElementType::Logical, vec![1, 1], vec![byte])).unwrap(),
            AssessmentFunctionOutcome {
                evaluated: true,
                accepted
            }
        );
    }
    for value in [
        dense(ElementType::Logical, vec![1, 2], vec![0, 1]),
        dense(ElementType::Logical, vec![1, 1], vec![2]),
        dense(ElementType::Logical, vec![1, 1], vec![]),
        dense(ElementType::U8, vec![1, 1], vec![1]),
    ] {
        assert!(decode(value).is_err());
    }
}
