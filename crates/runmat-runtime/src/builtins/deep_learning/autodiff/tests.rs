use super::tape::{Tape, TapeActivation, TapeFuture};
use super::*;
use runmat_value::{IntValue, IntegerStorage, StructArray};
use std::collections::HashMap;

#[test]
fn suspended_and_dropped_callback_futures_do_not_publish_their_tapes() {
    let first_tape = Rc::new(RefCell::new(Tape::new()));
    let second_tape = Rc::new(RefCell::new(Tape::new()));
    let make_future = |tape: Rc<RefCell<Tape>>| TapeFuture {
        tape,
        future: Some(Box::pin(async {
            assert!(tape_is_active());
            push_node(Tensor::new(vec![1.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
            futures::future::pending::<()>().await;
        })),
    };
    let mut first = Box::pin(make_future(first_tape.clone()));
    let mut second = Box::pin(make_future(second_tape.clone()));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(first.as_mut().poll(&mut context).is_pending());
    assert!(!tape_is_active());
    assert!(second.as_mut().poll(&mut context).is_pending());
    assert!(!tape_is_active());
    assert_eq!(first_tape.borrow().nodes.len(), 1);
    assert_eq!(second_tape.borrow().nodes.len(), 1);
    drop(first);
    drop(second);
    assert!(!tape_is_active());
}

#[test]
fn nested_tape_activation_restores_the_outer_graph() {
    let outer = Rc::new(RefCell::new(Tape::new()));
    let inner = Rc::new(RefCell::new(Tape::new()));
    let _outer = TapeActivation::enter(outer.clone());
    push_node(Tensor::new(vec![1.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
    {
        let _inner = TapeActivation::enter(inner.clone());
        push_node(Tensor::new(vec![2.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
    }
    push_node(Tensor::new(vec![3.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
    assert_eq!(outer.borrow().nodes.len(), 2);
    assert_eq!(inner.borrow().nodes.len(), 1);
}

#[test]
fn traced_node_ids_are_exact_and_belong_only_to_the_active_tape() {
    let outer = Rc::new(RefCell::new(Tape::new()));
    let inner = Rc::new(RefCell::new(Tape::new()));
    let _outer = TapeActivation::enter(outer.clone());
    let traced = annotate_dlarray_value(dlarray(4.0)).expect("outer traced value");
    assert_eq!(dlarray_node_id(&traced), Some(0));
    let Value::Object(object) = &traced else {
        panic!("dlarray")
    };
    assert_eq!(
        object.properties.get(AD_NODE_PROPERTY),
        Some(&Value::Int(IntValue::U64(0)))
    );
    {
        let _inner = TapeActivation::enter(inner.clone());
        assert_eq!(dlarray_node_id(&traced), None);
        let retraced = annotate_dlarray_value(traced.clone()).expect("inner leaf");
        assert_eq!(dlarray_node_id(&retraced), Some(0));
        assert_eq!(inner.borrow().nodes.len(), 1);
    }
    assert_eq!(dlarray_node_id(&traced), Some(0));
    assert_eq!(outer.borrow().nodes.len(), 1);

    let mut forged = traced;
    let Value::Object(object) = &mut forged else {
        panic!("dlarray")
    };
    object
        .properties
        .insert(AD_NODE_PROPERTY.into(), Value::Num(0.0));
    assert_eq!(dlarray_node_id(&forged), None);
    object
        .properties
        .insert(AD_NODE_PROPERTY.into(), Value::Int(IntValue::U64(u64::MAX)));
    assert_eq!(dlarray_node_id(&forged), None);
}

fn wrapped(data: Value) -> Value {
    let mut object = ObjectInstance::new(DLARRAY_CLASS.to_string());
    object.properties.insert("Data".into(), data);
    Value::Object(object)
}

fn dlarray(value: f64) -> Value {
    wrapped(Value::Num(value))
}

fn element(value: f64) -> StructValue {
    let mut structure = StructValue::new();
    structure.insert("payload", dlarray(value));
    structure
}

#[test]
fn annotation_and_gradient_preserve_typed_struct_array_shape_and_schema() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let input =
        Value::StructArray(StructArray::new(vec![element(1.0), element(2.0)], vec![2, 1]).unwrap());
    let annotated = annotate_tree(input).expect("annotate structure array");
    let gradient =
        gradient_tree_for_target(&annotated, &HashMap::new()).expect("gradient structure array");

    let Value::StructArray(gradient) = gradient else {
        panic!("expected typed structure array");
    };
    assert_eq!(gradient.shape(), &[2, 1]);
    assert_eq!(
        gradient
            .field_names()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["payload"]
    );
    for value in gradient.field_values("payload").unwrap() {
        let Value::Object(object) = value else {
            panic!("expected gradient dlarray");
        };
        let Value::Tensor(data) = object.properties.get("Data").unwrap() else {
            panic!("expected gradient tensor");
        };
        assert_eq!(data.materialize_f64(), [0.0]);
    }
}

#[test]
fn dlarray_constructor_records_active_tape_and_rejects_resident_tape_input() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident data");
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        let host = futures::executor::block_on(super::super::training::dlarray_builtin(
            Value::Num(2.0),
            vec![],
        ))
        .expect("host dlarray is tape-tracked");
        assert!(matches!(host, Value::Object(_)));
        assert_eq!(tape.borrow().nodes.len(), 1);

        provider.reset_telemetry();
        let error = futures::executor::block_on(super::super::training::dlarray_builtin(
            Value::GpuTensor(resident),
            vec![],
        ))
        .expect_err("resident autodiff needs a tape kernel");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn single_matrix_product_retains_owning_input_precision() {
    let left = wrapped(Value::Tensor(
        Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
    ));
    let right = wrapped(Value::Tensor(
        Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap(),
    ));
    let result = dlarray_mtimes_builtin(left, right).expect("single matrix product");
    let Value::Object(result) = result else {
        panic!("dlarray result")
    };
    let Value::Tensor(data) = result.properties.get("Data").unwrap() else {
        panic!("tensor result")
    };
    assert_eq!(data.shape, vec![2, 2]);
    assert_eq!(
        data.clone().into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![23.0, 34.0, 31.0, 46.0])
    );
}

#[test]
fn mixed_floating_arithmetic_is_symmetric_and_keeps_dlarray_metadata() {
    let mut single = ObjectInstance::new(DLARRAY_CLASS.to_string());
    single.properties.insert(
        "Data".into(),
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    );
    single
        .properties
        .insert("Format".into(), Value::String("CB".into()));
    single
        .properties
        .insert("Labels".into(), Value::String("CB".into()));
    let single = Value::Object(single);
    for output in [
        dlarray_plus_builtin(single.clone(), Value::Num(2.0)),
        dlarray_plus_builtin(Value::Num(2.0), single.clone()),
        dlarray_times_builtin(single.clone(), Value::Num(2.0)),
        dlarray_times_builtin(Value::Num(2.0), single.clone()),
        dlarray_minus_builtin(Value::Num(3.0), single.clone()),
        dlarray_rdivide_builtin(Value::Num(4.0), single.clone()),
    ] {
        let Value::Object(output) = output.expect("mixed arithmetic") else {
            panic!("dlarray output")
        };
        assert_eq!(
            output.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
        assert_eq!(
            output.properties.get("Labels"),
            Some(&Value::String("CB".into()))
        );
        let Some(Value::Tensor(data)) = output.properties.get("Data") else {
            panic!("dlarray data")
        };
        assert_eq!(data.numeric_dtype(), NumericDType::F32);
        assert_eq!(data.shape, vec![1, 2]);
    }
    let double = wrapped(Value::Tensor(
        Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap(),
    ));
    for output in [
        dlarray_plus_builtin(single.clone(), double.clone()),
        dlarray_plus_builtin(double, single),
    ] {
        let Value::Object(output) = output.expect("mixed arrays") else {
            panic!("dlarray output")
        };
        let Some(Value::Tensor(data)) = output.properties.get("Data") else {
            panic!("dlarray data")
        };
        assert_eq!(data.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn logical_dlarray_math_uses_explicit_double_boundary() {
    let input = wrapped(Value::Bool(true));
    let Value::Object(added) = dlarray_plus_builtin(input.clone(), Value::Num(2.0)).unwrap() else {
        panic!("dlarray output")
    };
    let Some(Value::Tensor(data)) = added.properties.get("Data") else {
        panic!("numeric data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F64);
    assert_eq!(data.materialize_f64(), [3.0]);
    let Value::Object(summed) = dlarray_sum_builtin(input, vec![]).unwrap() else {
        panic!("dlarray output")
    };
    let Some(Value::Tensor(data)) = summed.properties.get("Data") else {
        panic!("numeric data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F64);
    assert_eq!(data.materialize_f64(), [1.0]);
}

#[test]
fn method_output_arity_preflights_before_resident_operand_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident operand");
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(2));
        for result in [
            dlarray_plus_builtin(wrapped(Value::GpuTensor(resident.clone())), Value::Num(1.0)),
            dlarray_mtimes_builtin(wrapped(Value::GpuTensor(resident.clone())), Value::Num(1.0)),
            dlarray_sum_builtin(wrapped(Value::GpuTensor(resident)), vec![]),
        ] {
            let error = result.expect_err("one-output method");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        }
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn repeated_single_target_gradients_accumulate_without_class_mismatch() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let input = annotate_dlarray_value(wrapped(Value::Tensor(
        Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap(),
    )))
    .unwrap();
    let product = dlarray_times_builtin(input.clone(), input.clone()).unwrap();
    let loss = dlarray_sum_builtin(product, vec![Value::String("all".into())]).unwrap();
    let Value::Object(gradient) =
        futures::executor::block_on(dlgradient_builtin(loss, vec![input]))
            .expect("repeated single gradient")
    else {
        panic!("dlarray gradient")
    };
    let Some(Value::Tensor(data)) = gradient.properties.get("Data") else {
        panic!("gradient Data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F32);
    assert_eq!(data.materialize_f64(), [2.0, 4.0]);
}

#[test]
fn matrix_method_supports_scalar_multiplication_but_rejects_nd_matrices() {
    let mut right = ObjectInstance::new(DLARRAY_CLASS.to_string());
    right.properties.insert(
        "Data".into(),
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    );
    right
        .properties
        .insert("Format".into(), Value::String("CB".into()));
    let Value::Object(result) =
        dlarray_mtimes_builtin(Value::Num(3.0), Value::Object(right)).unwrap()
    else {
        panic!("dlarray result")
    };
    assert_eq!(
        result.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
    let Some(Value::Tensor(data)) = result.properties.get("Data") else {
        panic!("result Data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F32);
    assert_eq!(data.shape, vec![1, 2]);
    assert_eq!(data.materialize_f64(), [3.0, 6.0]);

    let nd = wrapped(Value::Tensor(
        Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap(),
    ));
    let matrix = wrapped(Value::Tensor(
        Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap(),
    ));
    let error = dlarray_mtimes_builtin(nd, matrix).expect_err("N-D matrix is not 2-D");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
}

#[test]
fn incompatible_elementwise_shapes_keep_owning_error_and_source() {
    let left = wrapped(Value::Tensor(
        Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
    ));
    let right = wrapped(Value::Tensor(
        Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
    ));
    for error in [
        dlarray_plus_builtin(left.clone(), right.clone()),
        dlarray_times_builtin(left.clone(), right.clone()),
        dlarray_rdivide_builtin(left.clone(), right.clone()),
    ] {
        let error = error.expect_err("shape mismatch");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(error.source.is_some(), "retain the tensor shape failure");
    }
}

#[test]
fn arithmetic_rejects_mutated_native_integer_data_without_rounding() {
    let large = u64::MAX;
    let integer_data = wrapped(Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap(),
    ));
    for result in [
        dlarray_plus_builtin(integer_data.clone(), dlarray(1.0)),
        dlarray_mtimes_builtin(integer_data.clone(), dlarray(1.0)),
        dlarray_sum_builtin(integer_data, vec![Value::String("all".into())]),
        dlarray_times_builtin(wrapped(Value::Int(IntValue::U64(large))), dlarray(1.0)),
    ] {
        let error = result.expect_err("integer payload must not widen to double");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    }
}

#[test]
fn gradient_output_arity_rejects_before_nested_provider_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident target");
        let nested = Value::Cell(CellArray::new(vec![Value::GpuTensor(resident)], 1, 1).unwrap());
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = futures::executor::block_on(dlgradient_builtin(dlarray(1.0), vec![nested]))
            .expect_err("one target cannot provide two outputs");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn gradient_does_not_turn_an_untraced_loss_into_a_new_leaf() {
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let target = annotate_dlarray_value(dlarray(2.0)).expect("traced target");
    let error = futures::executor::block_on(dlgradient_builtin(dlarray(3.0), vec![target]))
        .expect_err("loss must be produced on tape");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    assert_eq!(tape.borrow().nodes.len(), 1);
}

#[test]
fn gradient_nested_resident_target_rejects_without_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident target");
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let loss = annotate_dlarray_value(dlarray(1.0)).expect("traced loss");
        let nested = Value::Cell(CellArray::new(vec![Value::GpuTensor(resident)], 1, 1).unwrap());
        provider.reset_telemetry();
        let error = futures::executor::block_on(dlgradient_builtin(loss, vec![nested]))
            .expect_err("untraced resident target");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}
