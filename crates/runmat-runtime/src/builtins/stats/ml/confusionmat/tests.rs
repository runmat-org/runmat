use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::IntegerStorage;

#[test]
fn scalar_text_rejects_multirow_and_nd_labels_and_option_names() {
    for shape in [vec![2, 1], vec![1, 1, 2]] {
        let chars = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], shape).unwrap(),
        );
        let cell = Value::Cell(CellArray::new(vec![chars.clone()], 1, 1).unwrap());
        let error = block_on(confusionmat_builtin(
            None,
            cell.clone(),
            cell,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert!(error.message.contains("string scalar"));
        let error = block_on(confusionmat_builtin(
            None,
            Value::Num(1.),
            Value::Num(1.),
            crate::builtins::common::test_support::builtin_arguments(vec![chars, Value::Num(1.)]),
        ))
        .unwrap_err();
        assert!(error.message.contains("string scalar"));
    }
}

#[test]
fn gathered_labels_do_not_release_borrowed_provider_inputs() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1., 2.],
                shape: &[2, 1],
            })
            .unwrap();
        confusionmat(
            Value::GpuTensor(handle.clone()),
            Value::GpuTensor(handle.clone()),
            vec![],
            2,
        );
        assert!(block_on(provider.download(&handle)).is_ok());
    });
}

#[test]
fn discarded_outputs_still_validate_labels_and_options() {
    with_outputs(0, || {
        assert!(block_on(confusionmat_builtin(
            None,
            Value::Num(1.),
            Value::from("a"),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .is_err());
        assert!(block_on(confusionmat_builtin(
            None,
            Value::Num(1.),
            Value::Num(1.),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("bad"),
                Value::Num(1.)
            ])
        ))
        .is_err());
        assert!(
            matches!(block_on(confusionmat_builtin(None, Value::Num(1.),Value::Num(1.),crate::builtins::common::test_support::builtin_arguments(vec![]))).unwrap(),Value::OutputList(values) if values.is_empty())
        );
    });
}

#[test]
fn character_pages_are_not_silently_ignored() {
    let chars = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
    );
    let error = block_on(confusionmat_builtin(
        None,
        chars.clone(),
        chars,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(error.message.contains("two-dimensional"));
}

#[test]
fn signed_zero_labels_share_one_count_key() {
    let values = confusionmat(
        tensor(vec![-0., 0.], vec![2, 1]),
        tensor(vec![0., -0.], vec![2, 1]),
        vec![],
        2,
    );
    let Value::Tensor(counts) = &values[0] else {
        panic!("counts")
    };
    assert_eq!(counts.materialize_f64(), vec![2.]);
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
}

fn with_outputs<T>(count: usize, f: impl FnOnce() -> T) -> T {
    let _guard = crate::output_count::push_output_count(Some(count)).expect("output-count scope");
    f()
}

fn confusionmat(group: Value, grouphat: Value, rest: Vec<Value>, outputs: usize) -> Vec<Value> {
    with_outputs(outputs, || {
        let Value::OutputList(values) = block_on(confusionmat_builtin(
            None,
            group,
            grouphat,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect("confusionmat") else {
            panic!("expected output list");
        };
        values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list")
    })
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn confusionmat_accepts_rank_zero_and_rank_one_numeric_labels() {
    let scalar = confusionmat(
        tensor(vec![1.0], Vec::new()),
        tensor(vec![1.0], Vec::new()),
        Vec::new(),
        1,
    );
    let Value::Tensor(scalar) = &scalar[0] else {
        panic!("scalar confusion matrix");
    };
    assert_eq!(scalar.shape(), &[1, 1]);
    assert_eq!(scalar.materialize_f64(), vec![1.0]);

    let vector = confusionmat(
        tensor(vec![1.0, 2.0], vec![2]),
        tensor(vec![1.0, 2.0], vec![2]),
        Vec::new(),
        1,
    );
    let Value::Tensor(vector) = &vector[0] else {
        panic!("vector confusion matrix");
    };
    assert_eq!(vector.shape(), &[2, 2]);
    assert_eq!(vector.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn counts_numeric_labels_in_sorted_order() {
    let values = confusionmat(
        tensor(vec![2.0, 1.0, 2.0, 3.0], vec![4, 1]),
        tensor(vec![2.0, 2.0, 1.0, 3.0], vec![4, 1]),
        Vec::new(),
        2,
    );
    let Value::Tensor(c) = &values[0] else {
        panic!("matrix");
    };
    assert_eq!(c.shape(), &[3, 3]);
    assert_eq!(
        c.materialize_f64(),
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
    let Value::Tensor(order) = &values[1] else {
        panic!("order");
    };
    assert_eq!(order.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn typed_integer_labels_cover_all_classes_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let cases = [
        IntegerStorage::I8(vec![2, 1, 2]),
        IntegerStorage::I16(vec![2, 1, 2]),
        IntegerStorage::I32(vec![2, 1, 2]),
        IntegerStorage::I64(vec![2, 1, 2]),
        IntegerStorage::U8(vec![2, 1, 2]),
        IntegerStorage::U16(vec![2, 1, 2]),
        IntegerStorage::U32(vec![2, 1, 2]),
        IntegerStorage::U64(vec![2, 1, 2]),
    ];
    for storage in cases {
        let values = confusionmat(
            int_tensor(storage.clone(), vec![3, 1]),
            int_tensor(storage.clone(), vec![3, 1]),
            Vec::new(),
            2,
        );
        let Value::Tensor(c) = &values[0] else {
            panic!("matrix");
        };
        assert_eq!(c.materialize_f64(), vec![1.0, 0.0, 0.0, 2.0]);
        let Value::Tensor(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(
            order.integer_storage().map(IntegerStorage::numeric_dtype),
            Some(storage.numeric_dtype())
        );
        assert_eq!(order.integer_storage().unwrap().exact_values().len(), 2);
    }
}

#[test]
fn typed_integer_labels_reject_mixed_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = block_on(confusionmat_builtin(
        None,
        int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
        int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("label types must be compatible"));
}

#[test]
fn uint64_labels_above_flintmax_remain_distinct() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let a = 9_007_199_254_740_993_u64;
    let b = u64::MAX;
    let values = confusionmat(
        int_tensor(IntegerStorage::U64(vec![b, a, b]), vec![3, 1]),
        int_tensor(IntegerStorage::U64(vec![a, a, b]), vec![3, 1]),
        Vec::new(),
        2,
    );
    let Value::Tensor(c) = &values[0] else {
        panic!("matrix");
    };
    assert_eq!(c.materialize_f64(), vec![1.0, 1.0, 0.0, 1.0]);
    let Value::Tensor(order) = &values[1] else {
        panic!("order");
    };
    assert_eq!(
        order.integer_storage(),
        Some(&IntegerStorage::U64(vec![a, b]))
    );
}

#[test]
fn typed_integer_roles_are_independently_compatibility_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let group_err = block_on(confusionmat_builtin(
        None,
        Value::Int(IntValue::I8(1)),
        Value::Int(IntValue::I8(1)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        group_err.identifier(),
        CONFUSIONMAT_INTEGER_GROUP_EXTENSION.error_identifier
    );
    let grouphat_err = block_on(confusionmat_builtin(
        None,
        Value::Num(1.0),
        Value::Int(IntValue::I8(1)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        grouphat_err.identifier(),
        CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION.error_identifier
    );
    let order_err = block_on(confusionmat_builtin(
        None,
        Value::Num(1.0),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Order"),
            Value::Int(IntValue::I8(1)),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        order_err.identifier(),
        CONFUSIONMAT_INTEGER_ORDER_EXTENSION.error_identifier
    );
}

#[test]
fn resident_integer_labels_gate_before_gather_and_remain_exact_in_runmat_mode() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .unwrap();
        let group = gpu_helpers::upload_tensor(provider, &tensor).expect("group upload");
        let grouphat = gpu_helpers::upload_tensor(provider, &tensor).expect("grouphat upload");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = block_on(confusionmat_builtin(
                None,
                Value::GpuTensor(group.clone()),
                Value::GpuTensor(grouphat.clone()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap_err();
            assert_eq!(
                err.identifier(),
                CONFUSIONMAT_INTEGER_GROUP_EXTENSION.error_identifier
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let values = confusionmat(
            Value::GpuTensor(group),
            Value::GpuTensor(grouphat),
            Vec::new(),
            2,
        );
        let Value::Tensor(order) = &values[1] else {
            panic!("order");
        };
        assert_eq!(order.integer_storage(), tensor.integer_storage());
    });
}

#[test]
fn resident_floating_labels_use_the_resident_extension_gate() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let data = tensor.materialize_f64();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: tensor.shape(),
            })
            .expect("upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(confusionmat_builtin(
            None,
            Value::GpuTensor(handle.clone()),
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            CONFUSIONMAT_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    });
}

#[test]
fn honors_explicit_order_and_ignores_missing_labels() {
    let values = confusionmat(
        tensor(vec![1.0, f64::NAN, 2.0, 3.0], vec![4, 1]),
        tensor(vec![2.0, 1.0, 2.0, 1.0], vec![4, 1]),
        vec![
            Value::from("Order"),
            tensor(vec![3.0, 2.0, 1.0], vec![3, 1]),
        ],
        2,
    );
    let Value::Tensor(c) = &values[0] else {
        panic!("matrix");
    };
    assert_eq!(c.shape(), &[3, 3]);
    assert_eq!(
        c.materialize_f64(),
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    );
    let Value::Tensor(order) = &values[1] else {
        panic!("order");
    };
    assert_eq!(order.materialize_f64(), vec![3.0, 2.0, 1.0]);
}

#[test]
fn supports_string_and_cell_text_labels() {
    let group = Value::StringArray(
        StringArray::from_elements_unbounded(
            vec![
                "dog".into(),
                "cat".into(),
                "dog".into(),
                runmat_value::StringElement::Missing,
            ],
            vec![4, 1],
        )
        .unwrap(),
    );
    let predicted = Value::Cell(
        CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("dog")),
                Value::CharArray(CharArray::new_row("dog")),
                Value::CharArray(CharArray::new_row("cat")),
                Value::CharArray(CharArray::new_row("cat")),
            ],
            4,
            1,
        )
        .unwrap(),
    );
    let values = confusionmat(group, predicted, Vec::new(), 2);
    let Value::Tensor(c) = &values[0] else {
        panic!("matrix");
    };
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.materialize_f64(), vec![1.0, 1.0, 1.0, 0.0]);
    let Value::StringArray(order) = &values[1] else {
        panic!("order");
    };
    assert_eq!(order.data(), vec!["dog", "cat"]);
}

#[test]
fn inferred_logical_order_contains_false_and_true() {
    let values = confusionmat(Value::Bool(true), Value::Bool(true), Vec::new(), 2);
    let Value::Tensor(c) = &values[0] else {
        panic!("matrix");
    };
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.materialize_f64(), vec![0.0, 0.0, 0.0, 1.0]);
    let Value::LogicalArray(order) = &values[1] else {
        panic!("order");
    };
    assert_eq!(order.data, vec![0, 1]);
}

#[test]
fn explicit_order_must_cover_all_observed_labels() {
    let err = block_on(confusionmat_builtin(
        None,
        tensor(vec![1.0, 2.0], vec![2, 1]),
        tensor(vec![1.0, 3.0], vec![2, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Order"),
            tensor(vec![1.0, 2.0], vec![2, 1]),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:confusionmat:InvalidArgument")
    );
    assert!(err
        .message
        .contains("Order must contain all nonmissing labels"));
}

#[test]
fn supports_categorical_labels_in_category_order() {
    let categories = Value::StringArray(
        StringArray::new(
            vec!["low".into(), "medium".into(), "high".into()],
            vec![1, 3],
        )
        .unwrap(),
    );
    let group = crate::builtins::table::categorical_from_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::StringArray(
                StringArray::new(vec!["high".into(), "low".into(), "high".into()], vec![3, 1])
                    .unwrap(),
            ),
            categories.clone(),
        ]),
    )
    .unwrap();
    let predicted = crate::builtins::table::categorical_from_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![3, 1],
                )
                .unwrap(),
            ),
            categories,
        ]),
    )
    .unwrap();

    let values = confusionmat(group, predicted, Vec::new(), 2);
    let Value::Tensor(c) = &values[0] else {
        panic!("matrix");
    };
    assert_eq!(c.shape(), &[3, 3]);
    assert_eq!(
        c.materialize_f64(),
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    let Value::Object(order) = &values[1] else {
        panic!("categorical order");
    };
    assert!(order.is_class(runmat_types::standard::CATEGORICAL));
    let order_labels =
        crate::builtins::table::categorical_labels(&Value::Object(order.clone())).unwrap();
    assert_eq!(order_labels, vec!["low", "medium", "high"]);
}

#[test]
fn rejects_bad_shapes_and_options() {
    let err = block_on(confusionmat_builtin(
        None,
        tensor(vec![1.0, 2.0], vec![2, 1]),
        tensor(vec![1.0], vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:confusionmat:InvalidArgument")
    );

    let err = block_on(confusionmat_builtin(
        None,
        tensor(vec![1.0], vec![1, 1]),
        tensor(vec![1.0], vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Rows"),
            Value::from("all"),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("unsupported option"));

    let err = block_on(confusionmat_builtin(
        None,
        Value::StringArray(
            StringArray::new(
                vec!["a".into(), "b".into(), "c".into(), "d".into()],
                vec![2, 2],
            )
            .unwrap(),
        ),
        Value::StringArray(
            StringArray::new(
                vec!["a".into(), "b".into(), "c".into(), "d".into()],
                vec![2, 2],
            )
            .unwrap(),
        ),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("must be a vector"));
}
