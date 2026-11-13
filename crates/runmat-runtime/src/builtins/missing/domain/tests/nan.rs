use super::super::entrypoints::{
    nanmean_builtin, nanmedian_builtin, nanmin_builtin, nanstd_builtin,
    nansum_builtin as nansum_owned, nanvar_builtin,
};
use super::*;

async fn nansum_builtin(
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    nansum_owned(
        None,
        observation,
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
    .map(|owned| owned.value().clone())
}

#[test]
fn nanmean_alias_uses_omitnan() {
    let result = block_on(nanmean_builtin(
        tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("all")]),
    ))
    .unwrap();
    assert!(matches!(result, Value::Num(n) if (n - 2.0).abs() < 1e-12));
}

#[test]
fn nanmean_alias_imposes_omission_without_synthetic_argument() {
    let input = tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]);
    let result = block_on(nanmean_builtin(
        input.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("all"),
            Value::from("includenan"),
        ]),
    ))
    .expect("legacy alias retains its omission policy");
    assert!(matches!(result, Value::Num(value) if (value - 2.0).abs() < 1e-12));

    let ordinary = block_on(crate::builtins::math::reduction::mean::mean_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("all"),
            Value::from("includenan"),
        ]),
    ))
    .expect("ordinary mean honors the explicit inclusion policy");
    assert!(matches!(ordinary, Value::Num(value) if value.is_nan()));
}

#[test]
fn nan_aliases_impose_typed_omission_after_owner_controls() {
    type Invoke = fn(Value, Vec<Value>) -> BuiltinResult<Value>;
    let cases: [(&str, Invoke, f64); 5] = [
        (
            "nansum",
            |value, rest| block_on(nansum_builtin(None, value, rest)),
            4.0,
        ),
        (
            "nanmedian",
            |value, rest| {
                block_on(nanmedian_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            2.0,
        ),
        (
            "nanstd",
            |value, rest| {
                block_on(nanstd_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            2.0_f64.sqrt(),
        ),
        (
            "nanvar",
            |value, rest| {
                block_on(nanvar_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            2.0,
        ),
        (
            "nanmin",
            |value, rest| {
                block_on(nanmin_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            1.0,
        ),
    ];
    for (name, invoke, expected) in cases {
        let result = invoke(
            tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]),
            vec![Value::from("all"), Value::from("includenan")],
        )
        .unwrap_or_else(|error| panic!("{name} should admit owner controls: {error}"));
        let actual = match result {
            Value::Num(value) => value,
            Value::Tensor(ref value) if value.host_buffer().len() == 1 => {
                value.materialize_f64()[0]
            }
            other => panic!("{name} returned non-scalar {other:?}"),
        };
        assert!(
            (actual - expected).abs() < 1e-12,
            "{name}: {actual} != {expected}"
        );
    }
}

#[test]
fn legacy_nan_reductions_gate_typed_integer_data_by_compatibility_mode() {
    type LegacyNanCase = (
        fn(Value, Vec<Value>) -> BuiltinResult<Value>,
        &'static str,
        &'static str,
    );
    let cases: [LegacyNanCase; 4] = [
        (
            |value, rest| {
                block_on(nanmean_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            "nanmean",
            "RunMat:compatibility:NanmeanTypedIntegerInputExtension",
        ),
        (
            |value, rest| block_on(nansum_builtin(None, value, rest)),
            "nansum",
            "RunMat:compatibility:NansumTypedIntegerInputExtension",
        ),
        (
            |value, rest| {
                block_on(nanmin_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            "nanmin",
            "RunMat:compatibility:NanminTypedIntegerInputExtension",
        ),
        (
            |value, rest| {
                block_on(nanmedian_builtin(
                    value,
                    crate::builtins::common::test_support::builtin_arguments(rest),
                ))
            },
            "nanmedian",
            "RunMat:compatibility:NanmedianTypedIntegerInputExtension",
        ),
    ];

    for (invoke, name, identifier) in cases {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = invoke(Value::Int(IntValue::U16(7)), Vec::new()).unwrap_err();
            assert_eq!(error.identifier(), Some(identifier), "{name}");
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            invoke(Value::Int(IntValue::U16(7)), Vec::new())
                .unwrap_or_else(|error| panic!("{name} RunMat typed-integer input: {error}"));
        }
    }
}

#[test]
fn legacy_nan_reductions_gate_typed_integer_dimensions() {
    let floating = || tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let dimension = || Value::Int(IntValue::U8(2));
    let placeholder =
        || Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty placeholder"));

    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(nanmean_builtin(
        floating(),
        crate::builtins::common::test_support::builtin_arguments(vec![dimension()]),
    ))
    .expect_err("strict nanmean typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension")
    );
    let error = block_on(nansum_builtin(None, floating(), vec![dimension()]))
        .expect_err("strict nansum typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NansumTypedIntegerInputExtension")
    );
    let error = block_on(nanmedian_builtin(
        floating(),
        crate::builtins::common::test_support::builtin_arguments(vec![dimension()]),
    ))
    .expect_err("strict nanmedian typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NanmedianTypedIntegerInputExtension")
    );
    let error = block_on(nanmin_builtin(
        floating(),
        crate::builtins::common::test_support::builtin_arguments(vec![placeholder(), dimension()]),
    ))
    .expect_err("strict nanmin typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NanminTypedIntegerInputExtension")
    );
}

#[test]
fn legacy_nan_std_var_reject_integer_data_and_gate_integer_controls() {
    for enabled in [false, true] {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(enabled);
        let error = block_on(nanstd_builtin(
            Value::Int(IntValue::I16(7)),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(1),
            )]),
        ))
        .expect_err("nanstd integer data");
        assert!(error.message().contains("integer data inputs"));
        let error = block_on(nanvar_builtin(
            Value::Int(IntValue::I16(7)),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(1),
            )]),
        ))
        .expect_err("nanvar integer data");
        assert!(error.message().contains("integer data inputs"));
    }

    let floating = || tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(nanstd_builtin(
            floating(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(1),
            )]),
        ))
        .expect_err("nanstd strict typed-integer control");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanstdTypedIntegerControlExtension")
        );
        let error = block_on(nanvar_builtin(
            floating(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(1),
            )]),
        ))
        .expect_err("nanvar strict typed-integer control");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanvarTypedIntegerControlExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(nanstd_builtin(
            floating(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(1),
            )]),
        ))
        .expect("nanstd RunMat typed-integer control");
        block_on(nanvar_builtin(
            floating(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(1),
            )]),
        ))
        .expect("nanvar RunMat typed-integer control");
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn legacy_nan_integer_policy_inspects_resident_dtype_before_dispatch() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[1, u64::MAX]),
                shape: &[2, 1],
            })
            .expect("integer upload");

        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(nanmean_builtin(
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("strict resident nanmean");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension")
            );

            let error = block_on(nanstd_builtin(
                Value::GpuTensor(handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("resident nanstd integer data");
            assert!(error.message().contains("integer data inputs"));
        }

        handle
            .allocation_lease()
            .validate_live()
            .expect("borrowed input remains live");
        let physical = handle.allocation_lease().downgrade();
        handle.release().expect("release final fixture reference");
        assert!(
            physical.acquire().is_none(),
            "physical lease must expire after final release"
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn legacy_nan_integer_extensions_preserve_declared_residency() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        type ResidentNanCase = (
            &'static str,
            fn(Value, Vec<Value>) -> BuiltinResult<Value>,
            Vec<f64>,
        );
        let cases: [ResidentNanCase; 2] = [
            (
                "nanmean",
                |value, rest| {
                    block_on(nanmean_builtin(
                        value,
                        crate::builtins::common::test_support::builtin_arguments(rest),
                    ))
                },
                vec![2.0, 6.0],
            ),
            (
                "nansum",
                |value, rest| block_on(nansum_builtin(None, value, rest)),
                vec![4.0, 12.0],
            ),
        ];
        for (name, invoke, expected) in cases {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1, 3, 5, 7]),
                    shape: &[2, 2],
                })
                .expect("integer upload");
            let result =
                invoke(Value::GpuTensor(handle), Vec::new()).expect("resident integer extension");
            assert!(
                matches!(result, Value::GpuTensor(_)),
                "{name} returned {result:?}"
            );
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.materialize_f64(), expected);
        }

        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[1, 3, 5, 7]),
                shape: &[2, 2],
            })
            .expect("integer upload");
        let result = block_on(nanmedian_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("resident integer nanmedian");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather median");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![2, 6]))
        );
    });
}

#[test]
fn nanmin_supports_pairwise_form() {
    let result = block_on(nanmin_builtin(
        tensor(vec![f64::NAN, 4.0, 3.0], vec![1, 3]),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![2.0, f64::NAN, 5.0],
            vec![1, 3],
        )]),
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![2.0, 4.0, 3.0]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn nanmin_pairwise_preserves_rank_zero_scalarization_and_rank_one_broadcast_shape() {
    let scalar = block_on(nanmin_builtin(
        tensor(vec![f64::NAN], Vec::new()),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![2.0],
            Vec::new(),
        )]),
    ))
    .expect("rank-zero pairwise nanmin");
    assert!(matches!(scalar, Value::Num(value) if value == 2.0));

    let vector = block_on(nanmin_builtin(
        tensor(vec![f64::NAN], Vec::new()),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![2.0, 1.0],
            vec![2],
        )]),
    ))
    .expect("rank-zero scalar broadcasts over rank-one input");
    assert!(matches!(
        vector,
        Value::Tensor(tensor)
            if tensor.shape() == [2] && tensor.materialize_f64() == vec![2.0, 1.0]
    ));
}

#[test]
fn nanmin_empty_placeholder_selects_reduction_not_pairwise() {
    let input = tensor(vec![f64::NAN, 2.0, 3.0, 4.0], vec![2, 2]);
    let placeholder = Value::Tensor(
        Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty reduction placeholder"),
    );
    let result = block_on(nanmin_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![
            placeholder,
            Value::Num(2.0),
        ]),
    ))
    .expect("nanmin(A, [], dim) should reduce along dim");
    assert!(matches!(
        result,
        Value::Tensor(tensor)
            if tensor.shape() == [2, 1] && tensor.materialize_f64() == vec![3.0, 2.0]
    ));
}

#[test]
fn nanmin_all_option_routes_to_reduction() {
    let result = block_on(nanmin_builtin(
        tensor(vec![f64::NAN, 5.0, 2.0, 9.0], vec![2, 2]),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("all")]),
    ))
    .expect("nanmin(A, 'all') should reduce rather than compare pairwise");
    assert!(matches!(result, Value::Num(number) if number == 2.0));
}

#[test]
fn nanmin_reduction_returns_indices_but_pairwise_has_one_output() {
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let input = tensor(vec![f64::NAN, 2.0, 3.0, 4.0], vec![2, 2]);
        let placeholder = Value::Tensor(
            Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty reduction placeholder"),
        );
        let result = block_on(nanmin_builtin(
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![
                placeholder,
                Value::Num(2.0),
            ]),
        ))
        .expect("two-output reduction");
        let Value::OutputList(outputs) = result else {
            panic!("expected values and indices");
        };
        assert_eq!(outputs.len(), 2);
        assert!(
            matches!(&outputs[0], Value::Tensor(t) if t.shape() == [2, 1] && t.materialize_f64() == vec![3.0, 2.0])
        );
        assert!(
            matches!(&outputs[1], Value::Tensor(t) if t.shape() == [2, 1] && t.materialize_f64() == vec![2.0, 1.0])
        );
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(nanmin_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                vec![2.0, 1.0],
                vec![1, 2],
            )]),
        ))
        .expect_err("pairwise form has only one output");
        assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    }
}

#[test]
fn nanmin_pairwise_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let left = Tensor::new_integer(IntegerStorage::U16(vec![9, 4, 3]), vec![1, 3]).unwrap();
    let right = tensor(vec![2.0, f64::NAN, 5.0], vec![1, 3]);

    let result = block_on(nanmin_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![right]),
    ))
    .unwrap();

    assert!(
        matches!(result, Value::Tensor(tensor) if tensor.materialize_f64() == vec![2.0, 4.0, 3.0])
    );

    let left = tensor(vec![9.0, 4.0, 3.0], vec![1, 3]);
    let right = Tensor::new_integer(IntegerStorage::U8(vec![5]), vec![1, 1]).unwrap();

    let result = block_on(nanmin_builtin(
        left,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .unwrap();

    assert!(
        matches!(result, Value::Tensor(tensor) if tensor.materialize_f64() == vec![5.0, 4.0, 3.0])
    );
}

#[test]
fn nanmin_pairwise_preserves_exact_wide_same_class_integers() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let left = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        vec![1, 2],
    )
    .expect("left uint64");
    let right = Tensor::new_integer(
        IntegerStorage::U64(vec![1_u64 << 53, u64::MAX - 1]),
        vec![1, 2],
    )
    .expect("right uint64");

    let result = block_on(nanmin_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .unwrap();
    let Value::Tensor(tensor) = result else {
        panic!("expected uint64 tensor");
    };
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 53, u64::MAX - 1]))
    );
}

#[test]
fn nanmin_pairwise_never_falls_back_to_lossy_wide_integer_comparison() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mixed_classes = block_on(nanmin_builtin(
        Value::Int(IntValue::U64(u64::MAX)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I64(
            -1,
        ))]),
    ))
    .expect_err("mismatched integer classes cannot be widened through double");
    assert_eq!(
        mixed_classes.identifier(),
        Some("RunMat:missing:InvalidArgument")
    );
    assert!(mixed_classes.message().contains("matching integer classes"));

    let wide = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("wide integer tensor");
    let floating = tensor(vec![0.0, 1.0], vec![1, 2]);
    let mixed_array = block_on(nanmin_builtin(
        Value::Tensor(wide),
        crate::builtins::common::test_support::builtin_arguments(vec![floating]),
    ))
    .expect_err("inexact wide integer cannot enter floating-array fallback");
    assert_eq!(
        mixed_array.identifier(),
        Some("RunMat:missing:InvalidArgument")
    );
    assert!(mixed_array
        .message()
        .contains("cannot convert wide integers"));
}

#[test]
fn nanmin_pairwise_uses_omit_nan_for_exact_integer_and_nan_scalar() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = block_on(nanmin_builtin(
        Value::Int(IntValue::U64(u64::MAX)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(f64::NAN)]),
    ))
    .expect("NaN should select the exact integer operand");
    assert_eq!(result, Value::Int(IntValue::U64(u64::MAX)));
}

#[test]
fn nanmin_pairwise_propagates_exact_integer_shape_failure() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let left =
        Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![2, 1]).expect("integer left");
    let right =
        Tensor::new_integer(IntegerStorage::U64(vec![3, 4, 5]), vec![3, 1]).expect("integer right");
    let error = block_on(nanmin_builtin(
        Value::Tensor(left),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(right)]),
    ))
    .expect_err("incompatible exact integer shapes");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    assert!(error.message().contains("dimension mismatch"));
}
