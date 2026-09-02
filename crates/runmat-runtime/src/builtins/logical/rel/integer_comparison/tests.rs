use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{
    ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray, Tensor, Value,
};

fn array(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(runmat_value::Tensor::new_integer(storage, shape).expect("integer tensor"))
}

#[test]
fn compares_signed_unsigned_and_uint64_exactly() {
    let lhs = Value::Int(IntValue::U64(u64::MAX));
    let rhs = Value::Int(IntValue::I64(i64::MAX));
    assert_eq!(
        try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Gt).expect("comparison"),
        Some(Value::Bool(true))
    );
    assert_eq!(
        try_integer_comparison(
            &Value::Int(IntValue::I8(-1)),
            &Value::Int(IntValue::U8(0)),
            IntegerComparisonOp::Lt,
        )
        .expect("comparison"),
        Some(Value::Bool(true))
    );
}

#[test]
fn broadcasts_exact_integer_arrays_for_all_relations() {
    let lhs = array(IntegerStorage::U64(vec![0, u64::MAX]), vec![2, 1]);
    let rhs = array(IntegerStorage::I64(vec![0, 1, i64::MAX]), vec![1, 3]);
    let result = try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Ge)
        .expect("comparison")
        .expect("integer path");
    assert_eq!(
        result,
        Value::LogicalArray(
            LogicalArray::new(vec![1, 1, 0, 1, 0, 1], vec![2, 3]).expect("logical result")
        )
    );
}

#[test]
fn compares_integer_storage_to_scalar_double_without_64_bit_loss() {
    let exact = Value::Int(IntValue::U64((1_u64 << 53) + 1));
    let rounded = Value::Num((1_u64 << 53) as f64);
    assert_eq!(
        try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Eq).expect("comparison"),
        Some(Value::Bool(false))
    );
    assert_eq!(
        try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Gt).expect("comparison"),
        Some(Value::Bool(true))
    );

    let tensor = array(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![1, 2]);
    assert_eq!(
        try_integer_comparison(&tensor, &rounded, IntegerComparisonOp::Ne).expect("comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
        ))
    );
}

#[test]
fn compares_integer_storage_to_broadcast_float_arrays_without_64_bit_loss() {
    let integer = array(
        IntegerStorage::U64(vec![1_u64 << 53, (1_u64 << 53) + 1]),
        vec![2, 1],
    );
    let float = Value::Tensor(
        runmat_value::Tensor::new(
            vec![(1_u64 << 53) as f64, 0.0, (1_u64 << 53) as f64],
            vec![1, 3],
        )
        .expect("float tensor"),
    );
    let result = try_integer_comparison(&integer, &float, IntegerComparisonOp::Eq)
        .expect("comparison")
        .expect("integer path");
    assert_eq!(
        result,
        Value::LogicalArray(
            LogicalArray::new(vec![1, 0, 0, 0, 1, 0], vec![2, 3]).expect("logical result")
        )
    );

    let result = try_integer_comparison(&float, &integer, IntegerComparisonOp::Lt)
        .expect("comparison")
        .expect("integer path");
    assert_eq!(
        result,
        Value::LogicalArray(
            LogicalArray::new(vec![0, 1, 1, 1, 0, 1], vec![2, 3]).expect("logical result")
        )
    );
}

#[test]
fn compares_integer_storage_to_logical_arrays() {
    let integer = array(IntegerStorage::I8(vec![0, 1]), vec![1, 2]);
    let logical =
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array"));
    assert_eq!(
        try_integer_comparison(&integer, &logical, IntegerComparisonOp::Eq).expect("comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
        ))
    );
}

#[test]
fn compares_all_integer_storage_classes_to_complex_tensors_exactly() {
    let cases = [
        (
            IntegerStorage::I8(vec![-7, 5]),
            vec![(-7.0, 0.0), (0.0, 0.0)],
            vec![1, 0],
        ),
        (
            IntegerStorage::I16(vec![-300, 5]),
            vec![(-300.0, 0.0), (0.0, 0.0)],
            vec![1, 0],
        ),
        (
            IntegerStorage::I32(vec![-70_000, 5]),
            vec![(-70_000.0, 0.0), (0.0, 0.0)],
            vec![1, 0],
        ),
        (
            IntegerStorage::I64(vec![i64::MAX, -9_007_199_254_740_991]),
            vec![(i64::MAX as f64, 0.0), (-9_007_199_254_740_991.0, 0.0)],
            vec![0, 1],
        ),
        (
            IntegerStorage::U8(vec![7, 5]),
            vec![(7.0, 0.0), (0.0, 0.0)],
            vec![1, 0],
        ),
        (
            IntegerStorage::U16(vec![300, 5]),
            vec![(300.0, 0.0), (0.0, 0.0)],
            vec![1, 0],
        ),
        (
            IntegerStorage::U32(vec![70_000, 5]),
            vec![(70_000.0, 0.0), (0.0, 0.0)],
            vec![1, 0],
        ),
        (
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![((1_u64 << 53) as f64, 0.0), (u64::MAX as f64, 0.0)],
            vec![0, 0],
        ),
    ];

    for (storage, complex_data, expected_eq) in cases {
        let integer =
            runmat_value::Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
        let complex = Value::ComplexTensor(
            ComplexTensor::new(complex_data, vec![1, 2]).expect("complex tensor"),
        );
        let integer = Value::Tensor(integer);

        assert_eq!(
            try_complex_integer_equality_comparison(&integer, &complex, IntegerComparisonOp::Eq,)
                .expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(expected_eq.clone(), vec![1, 2]).expect("logical result")
            ))
        );
        assert_eq!(
            try_complex_integer_equality_comparison(&complex, &integer, IntegerComparisonOp::Ne,)
                .expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(
                    expected_eq.iter().map(|value| *value ^ 1).collect(),
                    vec![1, 2],
                )
                .expect("logical result")
            ))
        );
    }
}

#[test]
fn complex_ordering_uses_only_real_components_for_all_relations() {
    let lhs = Value::Complex(2.0, f64::NAN);
    let rhs = Value::Complex(2.0, f64::INFINITY);
    for (operation, expected) in [
        (IntegerComparisonOp::Lt, false),
        (IntegerComparisonOp::Le, true),
        (IntegerComparisonOp::Gt, false),
        (IntegerComparisonOp::Ge, true),
    ] {
        assert_eq!(
            try_complex_ordering_comparison(&lhs, &rhs, operation).expect("comparison"),
            Some(Value::Bool(expected))
        );
    }

    let nan_real = Value::Complex(f64::NAN, 0.0);
    for operation in [
        IntegerComparisonOp::Lt,
        IntegerComparisonOp::Le,
        IntegerComparisonOp::Gt,
        IntegerComparisonOp::Ge,
    ] {
        assert_eq!(
            try_complex_ordering_comparison(&nan_real, &Value::Num(0.0), operation)
                .expect("NaN comparison"),
            Some(Value::Bool(false))
        );
    }
    assert_eq!(
        try_complex_ordering_comparison(
            &Value::Complex(1.0, 0.0),
            &Value::Num(2.0),
            IntegerComparisonOp::Lt,
        )
        .expect("structurally complex zero-imaginary comparison"),
        Some(Value::Bool(true))
    );
}

#[test]
fn complex_integer_ordering_preserves_wide_real_components_and_broadcasts() {
    let storage = runmat_value::IntegerComplexStorage::new(
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        IntegerStorage::U64(vec![u64::MAX, 0]),
    )
    .expect("complex integer storage");
    let complex = Value::ComplexTensor(
        ComplexTensor::new_integer(storage, vec![2, 1]).expect("complex integer tensor"),
    );
    let rounded = Value::Tensor(
        Tensor::new(vec![(1_u64 << 53) as f64, u64::MAX as f64], vec![1, 2])
            .expect("double tensor"),
    );

    assert_eq!(
        try_complex_ordering_comparison(&complex, &rounded, IntegerComparisonOp::Gt)
            .expect("complex/double comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical result")
        ))
    );
    assert_eq!(
        try_complex_ordering_comparison(&rounded, &complex, IntegerComparisonOp::Lt)
            .expect("double/complex comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical result")
        ))
    );

    let rounded_complex = Value::ComplexTensor(
        ComplexTensor::new(
            vec![((1_u64 << 53) as f64, -1.0), (u64::MAX as f64, 1.0)],
            vec![1, 2],
        )
        .expect("floating complex tensor"),
    );
    assert_eq!(
        try_complex_ordering_comparison(&complex, &rounded_complex, IntegerComparisonOp::Gt,)
            .expect("integer-complex/floating-complex comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical result")
        ))
    );
}

#[test]
fn complex_single_and_double_ordering_broadcasts_without_using_imaginary_components() {
    let lhs = Value::ComplexTensor(
        ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.0, f32::NAN), (3.0, f32::INFINITY)].into()),
            vec![2, 1],
        )
        .expect("complex single"),
    );
    let rhs = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("double tensor"));
    let expected = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).expect("logical result");
    assert_eq!(
        try_complex_ordering_comparison(&lhs, &rhs, IntegerComparisonOp::Le)
            .expect("complex-single/double comparison"),
        Some(Value::LogicalArray(expected.clone()))
    );
    assert_eq!(
        try_complex_ordering_comparison(&rhs, &lhs, IntegerComparisonOp::Ge)
            .expect("double/complex-single comparison"),
        Some(Value::LogicalArray(expected))
    );
}

#[test]
fn complex_ordering_accepts_logical_and_character_numeric_operands() {
    let logical = Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical"));
    assert_eq!(
        try_complex_ordering_comparison(
            &Value::Complex(0.5, 99.0),
            &logical,
            IntegerComparisonOp::Gt,
        )
        .expect("complex/logical comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical result")
        ))
    );

    let chars = Value::CharArray(
        runmat_value::CharArray::new(vec!['A', 'C', 'B', 'D'], 2, 2).expect("character array"),
    );
    assert_eq!(
        try_complex_ordering_comparison(
            &Value::Complex(66.0, -99.0),
            &chars,
            IntegerComparisonOp::Lt,
        )
        .expect("complex/character comparison"),
        Some(Value::LogicalArray(
            LogicalArray::new(vec![0, 0, 1, 1], vec![2, 2]).expect("logical result")
        ))
    );
}

fn assert_resident_complex_ordering(provider: &dyn runmat_accelerate_api::AccelProvider) {
    let complex = ComplexTensor::new(vec![(1.0, 99.0), (3.0, -99.0)], vec![1, 2]).expect("complex");
    let complex_rhs =
        ComplexTensor::new(vec![(2.0, -7.0), (2.0, 7.0)], vec![1, 2]).expect("complex rhs");
    let real = Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("real");
    let complex_handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
    let complex_rhs_handle = gpu_helpers::upload_complex_tensor(provider, &complex_rhs).unwrap();
    let real_handle = gpu_helpers::upload_tensor(provider, &real).unwrap();
    for (builtin, expected, reverse_expected) in [
        ("lt", vec![1.0, 0.0], vec![0.0, 1.0]),
        ("le", vec![1.0, 0.0], vec![0.0, 1.0]),
        ("gt", vec![0.0, 1.0], vec![1.0, 0.0]),
        ("ge", vec![0.0, 1.0], vec![1.0, 0.0]),
    ] {
        for rhs in [&real_handle, &complex_rhs_handle] {
            let result = crate::call_builtin(
                builtin,
                &[
                    Value::GpuTensor(complex_handle.clone()),
                    Value::GpuTensor(rhs.clone()),
                ],
            )
            .expect("resident complex ordering");
            assert!(
                matches!(&result, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)),
                "{builtin} did not preserve logical residency for rhs storage {:?}: {result:?}",
                runmat_accelerate_api::handle_storage(rhs)
            );
            let gathered = test_support::gather(result).expect("gather logical result");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.materialize_f64(), expected);
        }
        let result = crate::call_builtin(
            builtin,
            &[
                Value::GpuTensor(real_handle.clone()),
                Value::GpuTensor(complex_handle.clone()),
            ],
        )
        .expect("reverse resident complex ordering");
        let gathered = test_support::gather(result).expect("gather reverse logical result");
        assert_eq!(gathered.materialize_f64(), reverse_expected);
    }
    let _ = provider.free(&complex_handle);
    let _ = provider.free(&complex_rhs_handle);
    let _ = provider.free(&real_handle);
}

#[test]
fn resident_complex_ordering_uses_provider_real_component_paths() {
    test_support::with_test_provider(assert_resident_complex_ordering);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_complex_ordering_uses_provider_real_component_paths() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    assert_resident_complex_ordering(provider);
}
