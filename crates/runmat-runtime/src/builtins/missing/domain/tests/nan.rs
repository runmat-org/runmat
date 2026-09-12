use super::super::entrypoints::{
    nanmean_builtin, nanmedian_builtin, nanmin_builtin, nanstd_builtin, nansum_builtin,
    nanvar_builtin,
};
use super::*;

#[test]
fn nanmean_alias_uses_omitnan() {
    let result = block_on(nanmean_builtin(
        tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]),
        vec![Value::from("all")],
    ))
    .unwrap();
    assert!(matches!(result, Value::Num(n) if (n - 2.0).abs() < 1e-12));
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
            |value, rest| block_on(nanmean_builtin(value, rest)),
            "nanmean",
            "RunMat:compatibility:NanmeanTypedIntegerInputExtension",
        ),
        (
            |value, rest| block_on(nansum_builtin(value, rest)),
            "nansum",
            "RunMat:compatibility:NansumTypedIntegerInputExtension",
        ),
        (
            |value, rest| block_on(nanmin_builtin(value, rest)),
            "nanmin",
            "RunMat:compatibility:NanminTypedIntegerInputExtension",
        ),
        (
            |value, rest| block_on(nanmedian_builtin(value, rest)),
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
    let error = block_on(nanmean_builtin(floating(), vec![dimension()]))
        .expect_err("strict nanmean typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension")
    );
    let error = block_on(nansum_builtin(floating(), vec![dimension()]))
        .expect_err("strict nansum typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NansumTypedIntegerInputExtension")
    );
    let error = block_on(nanmedian_builtin(floating(), vec![dimension()]))
        .expect_err("strict nanmedian typed-integer dimension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NanmedianTypedIntegerInputExtension")
    );
    let error = block_on(nanmin_builtin(floating(), vec![placeholder(), dimension()]))
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
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("nanstd integer data");
        assert!(error.message().contains("integer data inputs"));
        let error = block_on(nanvar_builtin(
            Value::Int(IntValue::I16(7)),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("nanvar integer data");
        assert!(error.message().contains("integer data inputs"));
    }

    let floating = || tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(nanstd_builtin(
            floating(),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("nanstd strict typed-integer control");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanstdTypedIntegerControlExtension")
        );
        let error = block_on(nanvar_builtin(
            floating(),
            vec![Value::Int(IntValue::U8(1))],
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
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect("nanstd RunMat typed-integer control");
        block_on(nanvar_builtin(
            floating(),
            vec![Value::Int(IntValue::U8(1))],
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
                Vec::new(),
            ))
            .expect_err("strict resident nanmean");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:NanmeanTypedIntegerInputExtension")
            );

            let error = block_on(nanstd_builtin(Value::GpuTensor(handle.clone()), Vec::new()))
                .expect_err("resident nanstd integer data");
            assert!(error.message().contains("integer data inputs"));
        }

        provider.free(&handle).ok();
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
                |value, rest| block_on(nanmean_builtin(value, rest)),
                vec![2.0, 6.0],
            ),
            (
                "nansum",
                |value, rest| block_on(nansum_builtin(value, rest)),
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
        let result = block_on(nanmedian_builtin(Value::GpuTensor(handle), Vec::new()))
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
        vec![tensor(vec![2.0, f64::NAN, 5.0], vec![1, 3])],
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![2.0, 4.0, 3.0]));
}

#[test]
fn nanmin_pairwise_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let left = Tensor::new_integer(IntegerStorage::U16(vec![9, 4, 3]), vec![1, 3]).unwrap();
    let right = tensor(vec![2.0, f64::NAN, 5.0], vec![1, 3]);

    let result = block_on(nanmin_builtin(Value::Tensor(left), vec![right])).unwrap();

    assert!(
        matches!(result, Value::Tensor(tensor) if tensor.materialize_f64() == vec![2.0, 4.0, 3.0])
    );

    let left = tensor(vec![9.0, 4.0, 3.0], vec![1, 3]);
    let right = Tensor::new_integer(IntegerStorage::U8(vec![5]), vec![1, 1]).unwrap();

    let result = block_on(nanmin_builtin(left, vec![Value::Tensor(right)])).unwrap();

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
        vec![Value::Tensor(right)],
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
