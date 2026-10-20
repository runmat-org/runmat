use super::tape::{NodeValue, Tape, TapeActivation, TapeFuture, TapePollAuthority};
use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, StructArray};
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
fn selected_context_polls_and_drops_on_its_tape_without_leaking_to_siblings() {
    let execution = Rc::new(crate::execution::RuntimeExecutionService::new());
    let unrelated = crate::context::RuntimeContext::new(execution);
    let outer = Rc::new(RefCell::new(Tape::new()));
    let selected_tape = Rc::new(RefCell::new(Tape::new()));
    let retired_tape = Rc::downgrade(&selected_tape);
    let selected = unrelated
        .clone()
        .with_tape_poll_authority(TapePollAuthority::new(&selected_tape));
    let dropped_on_selected_tape = Rc::new(std::cell::Cell::new(false));
    struct DropWitness(Rc<std::cell::Cell<bool>>, Rc<RefCell<Tape>>);
    impl Drop for DropWitness {
        fn drop(&mut self) {
            self.0
                .set(active_tape().is_some_and(|active| Rc::ptr_eq(&active, &self.1)));
        }
    }
    let _outer = TapeActivation::enter(outer.clone());
    {
        let _unrelated = unrelated.enter();
        assert!(
            !tape_is_active(),
            "an unrelated context masks the caller tape"
        );
    }
    let witness = DropWitness(dropped_on_selected_tape.clone(), selected_tape.clone());
    let mut child = Box::pin(selected.scope(async move {
        let _witness = witness;
        assert!(tape_is_active());
        push_node(Tensor::new(vec![2.0], vec![1, 1]).unwrap(), NodeKind::Leaf).unwrap();
        futures::future::pending::<()>().await;
    }));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(child.as_mut().poll(&mut context).is_pending());
    assert!(Rc::ptr_eq(&active_tape().unwrap(), &outer));
    drop(child);
    assert!(dropped_on_selected_tape.get());
    assert!(Rc::ptr_eq(&active_tape().unwrap(), &outer));
    assert_eq!(outer.borrow().nodes.len(), 0);
    assert_eq!(selected_tape.borrow().nodes.len(), 1);
    drop(selected_tape);
    assert!(
        retired_tape.upgrade().is_none(),
        "retained context cannot own the graph"
    );
    {
        let _expired = selected.enter();
        assert!(
            !tape_is_active(),
            "an expired token must mask, never borrow, the outer tape"
        );
    }
    assert!(Rc::ptr_eq(&active_tape().unwrap(), &outer));
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
        object.properties().get(AD_NODE_PROPERTY),
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
    for id in [Value::Num(0.0), Value::Int(IntValue::U64(u64::MAX))] {
        let Value::Object(object) = &mut forged else {
            panic!("dlarray")
        };
        object
            .replace_dl_property(AD_NODE_PROPERTY, id)
            .expect("fixture property");
        assert_eq!(dlarray_node_id(&forged), None);
    }
}

#[test]
fn tape_identity_requires_original_dense_storage_class_and_shape() {
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let traced = annotate_dlarray_value(wrapped(Value::Tensor(tensor.clone()))).unwrap();
    assert_eq!(dlarray_node_id(&traced), Some(0));

    let mut reshaped = traced.clone();
    let Value::Object(object) = &mut reshaped else {
        unreachable!()
    };
    object
        .replace_dl_property(
            "Data",
            Value::Tensor(tensor.clone().reshape(vec![1, 2]).unwrap()),
        )
        .expect("fixture property");
    assert_eq!(dlarray_node_id(&reshaped), None);

    let mut replaced = traced.clone();
    let Value::Object(object) = &mut replaced else {
        unreachable!()
    };
    object
        .replace_dl_property(
            "Data",
            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        )
        .expect("fixture property");
    assert_eq!(dlarray_node_id(&replaced), None);

    let mut changed = traced.clone();
    let Value::Object(object) = &mut changed else {
        unreachable!()
    };
    let Some(Value::Tensor(current)) = object.fixture_dl_property_mut("Data") else {
        unreachable!()
    };
    current
        .set_numeric_assignment_at(0, NumericScalar::F32(9.0))
        .unwrap();
    assert_eq!(dlarray_node_id(&changed), None);
    assert_eq!(dlarray_node_id(&traced), Some(0));
    assert_eq!(tape.borrow().nodes.len(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_tape_gradient_preserves_source_shape_dtype_and_values() {
    for (shape, values, expected) in [
        (Vec::new(), vec![2.0_f32], vec![4.0_f64]),
        (vec![2], vec![2.0_f32, 3.0], vec![4.0_f64, 6.0]),
    ] {
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let input = annotate_dlarray_value(wrapped(Value::Tensor(
            Tensor::from_f32(values, shape.clone()).expect("low-rank input"),
        )))
        .expect("traced low-rank input");
        let squared = dlarray_times_builtin(input.clone(), input.clone()).expect("square input");
        let loss = dlarray_sum_builtin(
            squared,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        )
        .expect("scalar loss");
        let Value::Object(gradient) = futures::executor::block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![input]),
        ))
        .expect("low-rank gradient") else {
            panic!("expected gradient dlarray");
        };
        let Some(Value::Tensor(data)) = gradient.properties().get("Data") else {
            panic!("expected gradient tensor");
        };
        assert_eq!(data.shape(), shape.as_slice());
        assert_eq!(data.numeric_dtype(), NumericDType::F32);
        assert_eq!(data.materialize_f64(), expected);
    }
}

#[test]
fn scalar_and_logical_tape_sources_preserve_exact_provenance_without_payload_scans() {
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let negative_zero = annotate_dlarray_value(wrapped(Value::Num(-0.0))).unwrap();
    assert_eq!(dlarray_node_id(&negative_zero), Some(0));
    let mut positive_zero = negative_zero.clone();
    let Value::Object(object) = &mut positive_zero else {
        unreachable!()
    };
    object
        .replace_dl_property("Data", Value::Num(0.0))
        .expect("fixture property");
    assert_eq!(dlarray_node_id(&positive_zero), None);

    let nan = f64::from_bits(0x7ff8_0000_0000_0042);
    let traced_nan = annotate_dlarray_value(wrapped(Value::Num(nan))).unwrap();
    assert_eq!(dlarray_node_id(&traced_nan), Some(1));
    let mut other_nan = traced_nan.clone();
    let Value::Object(object) = &mut other_nan else {
        unreachable!()
    };
    object
        .replace_dl_property("Data", Value::Num(f64::from_bits(0x7ff8_0000_0000_0043)))
        .expect("fixture property");
    assert_eq!(dlarray_node_id(&other_nan), None);

    let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
    let traced_logical =
        annotate_dlarray_value(wrapped(Value::LogicalArray(logical.clone()))).unwrap();
    assert_eq!(dlarray_node_id(&traced_logical), Some(2));
    let mut changed_logical = traced_logical.clone();
    let Value::Object(object) = &mut changed_logical else {
        unreachable!()
    };
    object
        .replace_dl_property(
            "Data",
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap()),
        )
        .expect("fixture property");
    assert_eq!(dlarray_node_id(&changed_logical), None);
    let Value::Object(object) = &traced_logical else {
        unreachable!()
    };
    let Some(Value::LogicalArray(source)) = object.properties().get("Data") else {
        unreachable!()
    };
    assert!(source.data.shares_allocation_with(&logical.data));
    assert_eq!(tape.borrow().nodes.len(), 3);
}

#[test]
fn arithmetic_detaches_an_edited_data_tag_until_explicit_reannotation() {
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let mut stale = annotate_dlarray_value(dlarray(2.0)).unwrap();
    let Value::Object(object) = &mut stale else {
        unreachable!()
    };
    object
        .replace_dl_property("Data", Value::Num(3.0))
        .expect("fixture property");
    assert_eq!(dlarray_node_id(&stale), None);
    let detached = dlarray_plus_builtin(stale.clone(), dlarray(4.0)).unwrap();
    assert_eq!(dlarray_node_id(&detached), None);
    assert_eq!(tape.borrow().nodes.len(), 1);
    let retraced = annotate_dlarray_value(stale).unwrap();
    assert_eq!(dlarray_node_id(&retraced), Some(1));
    assert_eq!(tape.borrow().nodes.len(), 2);
}

fn wrapped(data: Value) -> Value {
    let mut object = ObjectInstance::new(DLARRAY_CLASS.to_string());
    object
        .replace_dl_property("Data", data)
        .expect("fixture property");
    Value::Object(object)
}

fn dlarray(value: f64) -> Value {
    wrapped(Value::Num(value))
}

fn element(value: f64) -> StructValue {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("payload", dlarray(value));
    structure.finish()
}

#[test]
fn annotation_and_gradient_preserve_typed_struct_array_shape_and_schema() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let input =
        Value::StructArray(StructArray::new(vec![element(1.0), element(2.0)], vec![2, 1]).unwrap());
    let alias = input.clone();
    let annotated = annotate_tree(input).expect("annotate structure array");
    let (Value::StructArray(original), Value::StructArray(updated)) = (&alias, &annotated) else {
        panic!("typed structure arrays")
    };
    assert_eq!(original.shape(), updated.shape());
    assert_ne!(
        original.field_values("payload").unwrap().as_ptr(),
        updated.field_values("payload").unwrap().as_ptr()
    );
    for value in original.field_values("payload").unwrap() {
        assert_eq!(dlarray_node_id(value), None);
    }
    for value in updated.field_values("payload").unwrap() {
        assert!(dlarray_node_id(value).is_some());
    }
    let gradient = super::gradient::gradient_tree_for_target(&annotated, &HashMap::new())
        .expect("gradient structure array");

    let Value::StructArray(gradient) = gradient else {
        panic!("expected typed structure array");
    };
    assert_eq!(gradient.shape(), &[2, 1]);
    assert_eq!(gradient.field_names().collect::<Vec<_>>(), ["payload"]);
    for value in gradient.field_values("payload").unwrap() {
        let Value::Object(object) = value else {
            panic!("expected gradient dlarray");
        };
        let Value::Tensor(data) = object.properties().get("Data").unwrap() else {
            panic!("expected gradient tensor");
        };
        assert_eq!(data.materialize_f64(), [0.0]);
    }
}

#[test]
fn cell_annotation_reuses_unique_backing_and_keeps_shared_inert_aliases() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let input = CellArray::new_with_shape(vec![Value::Num(9.0), dlarray(2.0)], vec![1, 2, 1])
        .expect("cell input");
    let original = input.data().as_ptr();
    let Value::Cell(annotated) = annotate_tree(Value::Cell(input)).expect("annotated cell") else {
        panic!("expected cell");
    };
    assert_eq!(annotated.data().as_ptr(), original);
    assert_eq!(annotated.shape(), &[1, 2, 1]);
    assert_eq!(dlarray_node_id(&annotated.data()[1]), Some(0));

    let inert = CellArray::new(vec![Value::Num(3.0)], 1, 1).expect("inert cell");
    let alias = inert.clone();
    let pointer = inert.data().as_ptr();
    let Value::Cell(unchanged) = annotate_tree(Value::Cell(inert)).expect("inert alias") else {
        panic!("expected cell");
    };
    assert_eq!(unchanged.data().as_ptr(), pointer);
    assert_eq!(alias.data().as_ptr(), pointer);

    let shared = CellArray::new(vec![dlarray(4.0)], 1, 1).expect("shared active Cell");
    let original = shared.clone();
    let Value::Cell(updated) =
        annotate_tree(Value::Cell(shared)).expect("shared annotation destination")
    else {
        panic!("Cell destination")
    };
    assert_ne!(original.data().as_ptr(), updated.data().as_ptr());
    assert_eq!(dlarray_node_id(&original.data()[0]), None);
    assert!(dlarray_node_id(&updated.data()[0]).is_some());
}

#[test]
fn repeated_target_gradients_share_cached_copy_on_write_allocation() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let target = annotate_dlarray_value(wrapped(Value::Tensor(
        Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap(),
    )))
    .expect("traced target");
    let target_id = dlarray_node_id(&target).expect("active node");
    let gradient = Tensor::from_f32(vec![1.0, 1.0], vec![2, 1]).unwrap();
    let gradients = HashMap::from([(target_id, gradient)]);
    let first = super::gradient::gradient_tree_for_target(&target, &gradients).unwrap();
    let second = super::gradient::gradient_tree_for_target(&target, &gradients).unwrap();
    fn data(value: &Value) -> &Tensor {
        let Value::Object(object) = value else {
            panic!("gradient object")
        };
        let Some(Value::Tensor(data)) = object.properties().get("Data") else {
            panic!("gradient Data")
        };
        data
    }
    assert!(data(&first)
        .host_buffer()
        .shares_allocation_with(data(&second).host_buffer()));
    assert_eq!(data(&first).numeric_dtype(), NumericDType::F32);
}

#[test]
fn dlarray_constructor_records_host_and_resident_tape_sources_without_gather() {
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
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("host dlarray is tape-tracked");
        assert!(matches!(host, Value::Object(_)));
        assert_eq!(tape.borrow().nodes.len(), 1);

        provider.reset_telemetry();
        let traced = futures::executor::block_on(super::super::training::dlarray_builtin(
            Value::GpuTensor(resident.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("resident dlarray is tape-tracked");
        assert_eq!(dlarray_node_id(&traced), Some(1));
        let Value::Object(object) = &traced else {
            panic!("resident dlarray object");
        };
        let Some(Value::GpuTensor(data)) = object.properties().get("Data") else {
            panic!("resident Data");
        };
        assert_eq!(data.view_identity(), resident.view_identity());
        assert_eq!(data.shape(), &[1, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(data),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        let nodes = tape.borrow();
        assert_eq!(nodes.nodes.len(), 2);
        let NodeValue::Resident(leaf) = &nodes.nodes[1].value else {
            panic!("resident tape leaf");
        };
        assert_eq!(leaf.view_identity(), resident.view_identity());
        assert_eq!(leaf.shape(), &[1, 1]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(resident.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn dlarray_constructor_rejects_complex_resident_tape_source_before_publication() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let complex = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F64(&[1.0, 2.0]),
                shape: &[1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            })
            .expect("complex resident data");
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        provider.reset_telemetry();
        let error = futures::executor::block_on(super::super::training::dlarray_builtin(
            Value::GpuTensor(complex.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("complex resident tape source must be rejected");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(error.to_string().contains("real single or double storage"));
        assert!(tape.borrow().nodes.is_empty());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(complex.allocation_lease().validate_live().is_ok());
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
    let Value::Tensor(data) = result.properties().get("Data").unwrap() else {
        panic!("tensor result")
    };
    assert_eq!(data.shape(), &[2, 2]);
    assert_eq!(
        data.clone().into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![23.0, 34.0, 31.0, 46.0])
    );
}

#[test]
fn mixed_floating_arithmetic_is_symmetric_and_keeps_dlarray_metadata() {
    let mut single = ObjectInstance::new(DLARRAY_CLASS.to_string());
    single
        .replace_dl_property(
            "Data",
            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        )
        .expect("fixture property");
    single
        .replace_dl_property("Format", Value::String("CB".into()))
        .expect("fixture property");
    single
        .replace_dl_property("Labels", Value::String("CB".into()))
        .expect("fixture property");
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
            output.properties().get("Format"),
            Some(&Value::String("CB".into()))
        );
        assert_eq!(
            output.properties().get("Labels"),
            Some(&Value::String("CB".into()))
        );
        let Some(Value::Tensor(data)) = output.properties().get("Data") else {
            panic!("dlarray data")
        };
        assert_eq!(data.numeric_dtype(), NumericDType::F32);
        assert_eq!(data.shape(), &[1, 2]);
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
        let Some(Value::Tensor(data)) = output.properties().get("Data") else {
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
    let Some(Value::Tensor(data)) = added.properties().get("Data") else {
        panic!("numeric data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F64);
    assert_eq!(data.materialize_f64(), [3.0]);
    let Value::Object(summed) = dlarray_sum_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .unwrap() else {
        panic!("dlarray output")
    };
    let Some(Value::Tensor(data)) = summed.properties().get("Data") else {
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
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        for result in [
            dlarray_plus_builtin(wrapped(Value::GpuTensor(resident.clone())), Value::Num(1.0)),
            dlarray_mtimes_builtin(wrapped(Value::GpuTensor(resident.clone())), Value::Num(1.0)),
            dlarray_sum_builtin(
                wrapped(Value::GpuTensor(resident)),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ),
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
    let loss = dlarray_sum_builtin(
        product,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("all".into())]),
    )
    .unwrap();
    let Value::Object(gradient) = futures::executor::block_on(dlgradient_direct(
        loss,
        crate::builtins::common::test_support::builtin_arguments(vec![input]),
    ))
    .expect("repeated single gradient") else {
        panic!("dlarray gradient")
    };
    let Some(Value::Tensor(data)) = gradient.properties().get("Data") else {
        panic!("gradient Data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F32);
    assert_eq!(data.materialize_f64(), [2.0, 4.0]);
}

#[test]
fn sum_all_accepts_character_and_string_text_with_typed_gradients() {
    for option in [
        Value::String("all".into()),
        Value::CharArray(runmat_value::CharArray::new_row("all")),
        Value::StringArray(runmat_value::StringArray::new(vec!["ALL".into()], vec![1, 1]).unwrap()),
    ] {
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let input = annotate_dlarray_value(wrapped(Value::Tensor(
            Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )))
        .unwrap();
        let squared = dlarray_times_builtin(input.clone(), input.clone()).unwrap();
        let loss = dlarray_sum_builtin(
            squared,
            crate::builtins::common::test_support::builtin_arguments(vec![option]),
        )
        .unwrap();
        let Value::Object(gradient) = futures::executor::block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![input]),
        ))
        .unwrap() else {
            panic!("dlarray gradient")
        };
        let Some(Value::Tensor(data)) = gradient.properties().get("Data") else {
            panic!("gradient Data")
        };
        assert_eq!(data.numeric_dtype(), NumericDType::F32);
        assert_eq!(data.shape(), &[1, 3]);
        assert_eq!(data.materialize_f64(), [2.0, 4.0, 6.0]);
    }
}

#[test]
fn matrix_method_supports_scalar_multiplication_but_rejects_nd_matrices() {
    let mut right = ObjectInstance::new(DLARRAY_CLASS.to_string());
    right
        .replace_dl_property(
            "Data",
            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        )
        .expect("fixture property");
    right
        .replace_dl_property("Format", Value::String("CB".into()))
        .expect("fixture property");
    let Value::Object(result) =
        dlarray_mtimes_builtin(Value::Num(3.0), Value::Object(right)).unwrap()
    else {
        panic!("dlarray result")
    };
    assert_eq!(
        result.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
    let Some(Value::Tensor(data)) = result.properties().get("Data") else {
        panic!("result Data")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F32);
    assert_eq!(data.shape(), &[1, 2]);
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
        dlarray_sum_builtin(
            integer_data,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ),
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
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = futures::executor::block_on(dlgradient_direct(
            dlarray(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![nested]),
        ))
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
    let error = futures::executor::block_on(dlgradient_direct(
        dlarray(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![target]),
    ))
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
        let error = futures::executor::block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![nested]),
        ))
        .expect_err("untraced resident target");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn gradient_rejects_traced_values_mutated_to_resident_data_without_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[2.0],
                shape: &[1, 1],
            })
            .expect("resident Data");
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let loss = annotate_dlarray_value(dlarray(1.0)).expect("traced loss");
        let mut target = annotate_dlarray_value(dlarray(2.0)).expect("traced target");
        let target_id = dlarray_node_id(&target).expect("target node");
        let Value::Object(object) = &mut target else {
            panic!("dlarray target")
        };
        object
            .replace_dl_property("Data", Value::GpuTensor(resident.clone()))
            .expect("fixture property");

        let mut computed = HashMap::new();
        computed.insert(target_id, Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        provider.reset_telemetry();
        let error = super::gradient::gradient_tree_for_target(&target, &computed)
            .expect_err("a cached gradient cannot bypass edited Data provenance");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));

        let Value::Object(object) = &mut target else {
            panic!("dlarray target")
        };
        object
            .replace_dl_property("Data", Value::String("invalid".into()))
            .expect("fixture property");
        let error = super::gradient::gradient_tree_for_target(&target, &computed)
            .expect_err("a cached gradient cannot bypass invalid Data rejection");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));

        let mut resident_loss = loss;
        let Value::Object(object) = &mut resident_loss else {
            panic!("dlarray loss")
        };
        object
            .replace_dl_property("Data", Value::GpuTensor(resident))
            .expect("fixture property");
        let error = futures::executor::block_on(dlgradient_direct(
            resident_loss,
            crate::builtins::common::test_support::builtin_arguments(vec![target]),
        ))
        .expect_err("a traced loss cannot bypass edited Data provenance");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}
