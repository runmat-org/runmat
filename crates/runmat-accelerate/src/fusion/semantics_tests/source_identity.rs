use super::*;

#[test]
fn invocation_input_overrides_stale_constant_for_domain_facts_and_checked_codegen() {
    use runmat_accelerate_api::scalar_program::ScalarNode;

    runmat_gc::gc_test_context(|| {
        let add = |left, right, output| FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![left, right],
            output: Some(output),
        };
        let mut domain = plan(
            vec![
                cast("atanh", 0, 2),
                add(0, 1, 3),
                cast("acosh", 3, 4),
                add(2, 4, 5),
            ],
            5,
        );
        domain.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(3)]);
        domain.const_values.insert(
            0,
            Value::Tensor(runmat_value::Tensor::from_f32(vec![-2.0; 3], vec![1, 3]).unwrap()),
        );
        domain.const_values.insert(1, Value::Num(-1.0));
        let actual = [
            Value::Tensor(runmat_value::Tensor::new(vec![0.25, 0.5, 0.75], vec![1, 3]).unwrap()),
            Value::Num(1.0),
        ];
        let resolved = resolve(&domain, &actual).expect("live input proves the real domain");
        assert_eq!(
            resolved.fact(0).unwrap().numeric().unwrap().class,
            NumericClass::Double
        );
        domain
            .generate_typed_wgsl(&[5], &resolved, ScalarType::F64)
            .expect("typed codegen uses admitted live input");

        let mut checked = plan(vec![add(0, 1, 2)], 2);
        checked.const_values.insert(0, Value::Num(-99.0));
        let resolved = resolve(&checked, &[Value::Num(2.0), Value::Num(3.0)]).unwrap();
        resolved
            .require_source_plan(&checked)
            .expect("shadowed source constant remains in the immutable plan identity");
        let mut altered_source = checked.clone();
        altered_source.const_values.insert(0, Value::Num(-98.0));
        assert!(resolved.require_source_plan(&altered_source).is_err());
        let receipt = crate::fusion::typed_producer::TypedProducer::emit_checked_program(
            &checked,
            &resolved,
            ScalarType::F64,
            2,
        )
        .unwrap()
        .expect("closed addition receipt");
        assert!(receipt
            .wire()
            .nodes
            .iter()
            .any(|node| matches!(node, ScalarNode::Input { id: 0, .. })));
        assert!(!receipt
            .wire()
            .nodes
            .iter()
            .any(|node| matches!(node, ScalarNode::Literal { id: 0, .. })));
        let shader = checked
            .generate_typed_wgsl(&[2], &resolved, ScalarType::F64)
            .unwrap();
        assert!(shader.contains("input0.data"), "{shader}");
    });
}

#[test]
fn duplicate_invocation_inputs_and_leaf_producer_collisions_fail_closed() {
    let add = |left, right, output| FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![left, right],
        output: Some(output),
    };
    let mut duplicate = plan(vec![add(0, 1, 2)], 2);
    duplicate.inputs = vec![0, 0];
    assert!(resolve(&duplicate, &[Value::Num(1.0), Value::Num(2.0)])
        .unwrap_err()
        .to_string()
        .contains("duplicated"));

    let mut collision = plan(vec![add(0, 1, 2)], 2);
    collision.const_values.insert(2, Value::Num(9.0));
    assert!(resolve(&collision, &[Value::Num(1.0), Value::Num(2.0)])
        .unwrap_err()
        .to_string()
        .contains("conflicting producers"));
}

#[test]
fn failed_leaf_validation_records_attempted_preparation_without_provider_work() {
    use crate::placement::{begin_trace, report, PlacementEventKind};

    let add = |output| FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![0, 1],
        output: Some(output),
    };
    let mut invalid = plan(vec![add(2), add(2)], 2);
    invalid.const_values.insert(0, Value::Num(-99.0));
    let values = [Value::Num(2.0), Value::Num(3.0)];
    let inputs: Vec<_> = values.iter().collect();
    let trace = begin_trace("fusion.invalid_leaf_index", None);
    let correlation = trace.correlation();
    let invalid_bindings = crate::fusion::semantic_bindings::capture(&invalid).unwrap();
    assert!(
        futures::executor::block_on(ResolvedProgram::resolve_captured(
            &invalid,
            &inputs,
            ProviderPrecision::F64,
            Some(correlation),
            &invalid_bindings,
        ))
        .is_err()
    );
    trace.complete(None, None);
    let placement = report();
    let events = &placement
        .traces
        .iter()
        .find(|trace| trace.correlation == correlation)
        .expect("correlated structural preparation")
        .events;
    let prepares: Vec<_> = events
        .iter()
        .filter(|event| event.kind == PlacementEventKind::Prepare)
        .collect();
    assert_eq!(prepares.len(), 1);
    let preparation = prepares[0];
    assert_eq!(
        preparation.reason.as_deref(),
        Some("fusion_leaf_index_wall")
    );
    assert!(preparation.duration_ns.is_some());
    for (key, value) in [
        ("completed", 0),
        ("input_ids_indexed", 2),
        ("source_constants_indexed", 1),
        ("producer_ids_checked", 2),
        ("unique_leaves", 2),
    ] {
        assert!(preparation
            .attributes
            .iter()
            .any(|attribute| attribute.key == key && attribute.value == value));
    }
    assert!(!events.iter().any(|event| matches!(
        event.kind,
        PlacementEventKind::Upload | PlacementEventKind::Kernel
    )));

    let mut valid = plan(vec![add(2)], 2);
    valid.const_values.insert(0, Value::Num(-99.0));
    let trace = begin_trace("fusion.valid_leaf_index", None);
    let correlation = trace.correlation();
    let valid_bindings = crate::fusion::semantic_bindings::capture(&valid).unwrap();
    futures::executor::block_on(ResolvedProgram::resolve_captured(
        &valid,
        &inputs,
        ProviderPrecision::F64,
        Some(correlation),
        &valid_bindings,
    ))
    .expect("valid source with one shadowed constant");
    trace.complete(None, None);
    let placement = report();
    let events = &placement
        .traces
        .iter()
        .find(|trace| trace.correlation == correlation)
        .expect("correlated successful structural preparation")
        .events;
    let preparation = events
        .iter()
        .find(|event| event.reason.as_deref() == Some("fusion_leaf_index_wall"))
        .expect("successful leaf index receipt");
    assert!(preparation
        .attributes
        .iter()
        .any(|attribute| attribute.key == "completed" && attribute.value == 1));
    assert!(preparation
        .attributes
        .iter()
        .any(|attribute| attribute.key == "unique_leaves" && attribute.value == 2));
}

#[test]
fn unbound_source_constant_remains_a_checked_literal() {
    use runmat_accelerate_api::scalar_program::ScalarNode;

    let mut source = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 3],
            output: Some(2),
        }],
        2,
    );
    source.inputs = vec![0];
    source.stack_pattern = vec![0];
    source.const_values.insert(3, Value::Num(4.0));
    let resolved = resolve(&source, &[Value::Num(2.0)]).unwrap();
    let receipt = crate::fusion::typed_producer::TypedProducer::emit_checked_program(
        &source,
        &resolved,
        ScalarType::F64,
        2,
    )
    .unwrap()
    .expect("closed source literal receipt");
    assert!(receipt
        .wire()
        .nodes
        .iter()
        .any(|node| matches!(node, ScalarNode::Input { id: 0, .. })));
    assert!(receipt
        .wire()
        .nodes
        .iter()
        .any(|node| matches!(node, ScalarNode::Literal { id: 3, .. })));
}

#[test]
fn equal_facts_do_not_merge_distinct_primitive_or_operand_topology() {
    let add = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let subtract = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Sub,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let reversed = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Sub,
            inputs: vec![1, 0],
            output: Some(2),
        }],
        2,
    );
    let values = [Value::Num(7.0), Value::Num(3.0)];
    let add_facts = resolve(&add, &values).unwrap();
    let subtract_facts = resolve(&subtract, &values).unwrap();
    let reversed_facts = resolve(&reversed, &values).unwrap();
    assert_eq!(add_facts.fact(2).unwrap(), subtract_facts.fact(2).unwrap());
    assert_ne!(
        add_facts.cache_identity().unwrap(),
        subtract_facts.cache_identity().unwrap()
    );
    assert_ne!(
        subtract_facts.cache_identity().unwrap(),
        reversed_facts.cache_identity().unwrap()
    );
    assert!(subtract_facts.require_source_plan(&reversed).is_err());
    assert!(subtract_facts.require_source_plan(&subtract).is_ok());
}

#[test]
fn source_constants_retain_signed_zero_values_and_nan_payloads() {
    let with_constant = |value: Value| {
        let mut source = plan(
            vec![FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![0, 2],
                output: Some(3),
            }],
            3,
        );
        source.inputs = vec![0];
        source.stack_pattern = vec![0];
        source.const_values.insert(2, value);
        source
    };
    let positive = with_constant(Value::Num(0.0));
    let negative = with_constant(Value::Num(-0.0));
    let one = with_constant(Value::Num(1.0));
    let two = with_constant(Value::Num(2.0));
    let input = [Value::Num(4.0)];
    let positive_facts = resolve(&positive, &input).unwrap();
    assert_ne!(
        positive_facts.cache_identity().unwrap(),
        resolve(&negative, &input)
            .unwrap()
            .cache_identity()
            .unwrap()
    );
    assert_ne!(
        resolve(&one, &input).unwrap().cache_identity().unwrap(),
        resolve(&two, &input).unwrap().cache_identity().unwrap()
    );
    assert!(positive_facts.require_source_plan(&negative).is_err());

    let nan_a = with_constant(Value::Num(f64::from_bits(0x7ff8_0000_0000_0001)));
    let nan_b = with_constant(Value::Num(f64::from_bits(0x7ff8_0000_0000_0002)));
    assert_ne!(
        resolve(&nan_a, &input).unwrap().cache_identity().unwrap(),
        resolve(&nan_b, &input).unwrap().cache_identity().unwrap()
    );
}

#[test]
fn slot_constants_are_authenticated_even_when_value_ids_are_unchanged() {
    let mut first = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    first.constants.insert(0, Value::Num(1.0));
    let mut second = first.clone();
    second.constants.insert(0, Value::Num(2.0));
    let values = [Value::Num(4.0), Value::Num(5.0)];
    let admitted = resolve(&first, &values).unwrap();
    assert_ne!(
        admitted.cache_identity().unwrap(),
        resolve(&second, &values).unwrap().cache_identity().unwrap()
    );
    assert!(admitted.require_source_plan(&second).is_err());
}

#[test]
fn tensor_constant_identity_retains_physical_precision_and_shape() {
    let source = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let with_payload = |tensor| {
        let mut plan = source.clone();
        plan.const_values.insert(7, Value::Tensor(tensor));
        plan
    };
    let double = with_payload(runmat_value::Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    let single = with_payload(runmat_value::Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
    let rank_three = with_payload(runmat_value::Tensor::new(vec![1.0], vec![1, 1, 1]).unwrap());
    let values = [Value::Num(4.0), Value::Num(5.0)];
    let admitted = resolve(&double, &values).unwrap();
    assert_ne!(
        admitted.cache_identity().unwrap(),
        resolve(&single, &values).unwrap().cache_identity().unwrap()
    );
    assert_ne!(
        admitted.cache_identity().unwrap(),
        resolve(&rank_three, &values)
            .unwrap()
            .cache_identity()
            .unwrap()
    );
    assert!(admitted.require_source_plan(&single).is_err());
    assert!(admitted.require_source_plan(&rank_three).is_err());
}

#[test]
fn opaque_plan_constant_fails_before_source_admission() {
    let mut source = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    source
        .const_values
        .insert(7, Value::function_handle("opaque".into()));
    let error = resolve(&source, &[Value::Num(4.0), Value::Num(5.0)]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("fusion constant has no exact source identity"),
        "{error}"
    );
}
