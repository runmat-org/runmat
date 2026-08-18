#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use runmat_builtins::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, SparseTensor,
    Tensor, Value,
};

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

fn expect_integer(value: Value, shape: &[usize], storage: IntegerStorage) {
    match value {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, shape);
            assert_eq!(tensor.integer_storage(), Some(&storage));
        }
        other => panic!("expected integer tensor, got {other:?}"),
    }
}

fn expect_integer_class(value: Value, class_name: &str) {
    match value {
        Value::Tensor(tensor) => assert_eq!(
            tensor
                .integer_storage()
                .expect("integer result storage")
                .class_name(),
            class_name
        ),
        Value::Int(value) => assert_eq!(value.class_name(), class_name),
        other => panic!("expected {class_name} result, got {other:?}"),
    }
}

fn expect_double_numeric(value: &Value) {
    match value {
        Value::Num(_) => {}
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), runmat_builtins::NumericDType::F64)
        }
        other => panic!("expected double numeric result, got {other:?}"),
    }
}

fn integer_set_classes() -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![1, 2, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 2, 3]),
    ]
}

fn expect_logical(value: Value, shape: &[usize], data: &[u8]) {
    assert_eq!(
        value,
        Value::LogicalArray(LogicalArray::new(data.to_vec(), shape.to_vec()).expect("logical"))
    );
}

fn expect_integer_logical_arithmetic_error(builtin: &str, lhs: Value, rhs: Value) {
    let error = runmat_runtime::call_builtin(builtin, &[lhs, rhs])
        .expect_err("integer/logical arithmetic must fail");
    assert!(
        error
            .to_string()
            .contains("integer arrays can only be combined with scalar double values"),
        "{builtin}: unexpected error: {error}"
    );
}

#[test]
fn registered_integer_arithmetic_rejects_ordered_logicals_for_every_class() {
    let integer_values = [
        IntValue::I8(2),
        IntValue::I16(2),
        IntValue::I32(2),
        IntValue::I64(2),
        IntValue::U8(2),
        IntValue::U16(2),
        IntValue::U32(2),
        IntValue::U64(2),
    ];
    let logical_values = [
        Value::Bool(true),
        Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).expect("scalar logical array")),
    ];
    for builtin in [
        "plus", "minus", "times", "rdivide", "ldivide", "power", "rem", "mod",
    ] {
        for integer in &integer_values {
            let integer_operands = [
                Value::Int(integer.clone()),
                integer_tensor(IntegerStorage::from_scalar(integer.clone()), vec![1, 1]),
                Value::SparseTensor(
                    SparseTensor::new_integer(
                        1,
                        1,
                        vec![0, 1],
                        vec![0],
                        IntegerStorage::from_scalar(integer.clone()),
                    )
                    .expect("scalar sparse integer"),
                ),
            ];
            for integer_operand in integer_operands {
                for logical_operand in &logical_values {
                    expect_integer_logical_arithmetic_error(
                        builtin,
                        integer_operand.clone(),
                        logical_operand.clone(),
                    );
                    expect_integer_logical_arithmetic_error(
                        builtin,
                        logical_operand.clone(),
                        integer_operand.clone(),
                    );
                }
            }
        }
    }
}

#[test]
fn registered_integer_power_rejects_invalid_exponents_for_every_class() {
    let bases = [
        IntValue::I8(2),
        IntValue::I16(2),
        IntValue::I32(2),
        IntValue::I64(2),
        IntValue::U8(2),
        IntValue::U16(2),
        IntValue::U32(2),
        IntValue::U64(2),
    ];
    for base in bases {
        for exponent in [-1.0, 0.5, f64::INFINITY, f64::NAN] {
            let error = runmat_runtime::call_builtin(
                "power",
                &[Value::Int(base.clone()), Value::Num(exponent)],
            )
            .expect_err("invalid integer exponent");
            assert_eq!(error.identifier(), Some("RunMat:power:InvalidInput"));
            assert!(error.message().contains("nonnegative integer values"));
        }
    }

    let signed_exponents = [
        integer_tensor(IntegerStorage::I8(vec![2, -1]), vec![1, 2]),
        integer_tensor(IntegerStorage::I16(vec![2, -1]), vec![1, 2]),
        integer_tensor(IntegerStorage::I32(vec![2, -1]), vec![1, 2]),
        integer_tensor(IntegerStorage::I64(vec![2, -1]), vec![1, 2]),
    ];
    let signed_bases = [
        integer_tensor(IntegerStorage::I8(vec![2]), vec![1, 1]),
        integer_tensor(IntegerStorage::I16(vec![2]), vec![1, 1]),
        integer_tensor(IntegerStorage::I32(vec![2]), vec![1, 1]),
        integer_tensor(IntegerStorage::I64(vec![2]), vec![1, 1]),
    ];
    for (base, exponent) in signed_bases.into_iter().zip(signed_exponents) {
        let error = runmat_runtime::call_builtin("power", &[base, exponent])
            .expect_err("negative signed integer exponent");
        assert_eq!(error.identifier(), Some("RunMat:power:InvalidInput"));
        assert!(error.message().contains("nonnegative integer values"));
    }
}

#[test]
fn registered_complex_integer_ordering_uses_exact_real_components_for_every_class() {
    let cases = [
        (
            IntegerStorage::I8(vec![0, 2]),
            IntegerStorage::I8(vec![i8::MAX, i8::MIN]),
        ),
        (
            IntegerStorage::I16(vec![0, 2]),
            IntegerStorage::I16(vec![i16::MAX, i16::MIN]),
        ),
        (
            IntegerStorage::I32(vec![0, 2]),
            IntegerStorage::I32(vec![i32::MAX, i32::MIN]),
        ),
        (
            IntegerStorage::I64(vec![0, 2]),
            IntegerStorage::I64(vec![i64::MAX, i64::MIN]),
        ),
        (
            IntegerStorage::U8(vec![0, 2]),
            IntegerStorage::U8(vec![u8::MAX, 0]),
        ),
        (
            IntegerStorage::U16(vec![0, 2]),
            IntegerStorage::U16(vec![u16::MAX, 0]),
        ),
        (
            IntegerStorage::U32(vec![0, 2]),
            IntegerStorage::U32(vec![u32::MAX, 0]),
        ),
        (
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 0]),
        ),
    ];
    for (real, imag) in cases {
        let is_wide = matches!(&real, IntegerStorage::U64(_));
        let complex = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(real, imag).expect("matching complex storage"),
                vec![1, 2],
            )
            .expect("complex integer tensor"),
        );
        let threshold = if is_wide {
            Value::Num((1_u64 << 53) as f64)
        } else {
            Value::Num(1.0)
        };
        for (builtin, expected) in [
            ("lt", [if is_wide { 0 } else { 1 }, 0]),
            ("le", [if is_wide { 0 } else { 1 }, 0]),
            ("gt", [if is_wide { 1 } else { 0 }, 1]),
            ("ge", [if is_wide { 1 } else { 0 }, 1]),
        ] {
            let result =
                runmat_runtime::call_builtin(builtin, &[complex.clone(), threshold.clone()])
                    .expect("registered complex integer comparison");
            expect_logical(result, &[1, 2], &expected);
        }
    }
}

#[test]
fn registered_integer_scalar_mtimes_preserves_every_class_and_saturates() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MAX, 2]),
            IntValue::I8(2),
            IntegerStorage::I8(vec![i8::MAX, 4]),
        ),
        (
            IntegerStorage::I16(vec![i16::MAX, 2]),
            IntValue::I16(2),
            IntegerStorage::I16(vec![i16::MAX, 4]),
        ),
        (
            IntegerStorage::I32(vec![i32::MAX, 2]),
            IntValue::I32(2),
            IntegerStorage::I32(vec![i32::MAX, 4]),
        ),
        (
            IntegerStorage::I64(vec![i64::MAX, 2]),
            IntValue::I64(2),
            IntegerStorage::I64(vec![i64::MAX, 4]),
        ),
        (
            IntegerStorage::U8(vec![u8::MAX, 2]),
            IntValue::U8(2),
            IntegerStorage::U8(vec![u8::MAX, 4]),
        ),
        (
            IntegerStorage::U16(vec![u16::MAX, 2]),
            IntValue::U16(2),
            IntegerStorage::U16(vec![u16::MAX, 4]),
        ),
        (
            IntegerStorage::U32(vec![u32::MAX, 2]),
            IntValue::U32(2),
            IntegerStorage::U32(vec![u32::MAX, 4]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1]),
            IntValue::U64(2),
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 54) + 2]),
        ),
    ];
    for (array, scalar, expected) in cases {
        let array = integer_tensor(array, vec![1, 2]);
        for operands in [
            [array.clone(), Value::Int(scalar.clone())],
            [Value::Int(scalar.clone()), array.clone()],
        ] {
            let result =
                runmat_runtime::call_builtin("mtimes", &operands).expect("integer scalar mtimes");
            expect_integer(result, &[1, 2], expected.clone());
        }
    }
}

#[test]
fn registered_integer_scalar_right_mrdivide_preserves_every_class() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MIN, 6]),
            IntValue::I8(2),
            IntegerStorage::I8(vec![i8::MIN / 2, 3]),
        ),
        (
            IntegerStorage::I16(vec![i16::MIN, 6]),
            IntValue::I16(2),
            IntegerStorage::I16(vec![i16::MIN / 2, 3]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, 6]),
            IntValue::I32(2),
            IntegerStorage::I32(vec![i32::MIN / 2, 3]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, 6]),
            IntValue::I64(2),
            IntegerStorage::I64(vec![i64::MIN / 2, 3]),
        ),
        (
            IntegerStorage::U8(vec![u8::MAX - 1, 6]),
            IntValue::U8(2),
            IntegerStorage::U8(vec![(u8::MAX - 1) / 2, 3]),
        ),
        (
            IntegerStorage::U16(vec![u16::MAX - 1, 6]),
            IntValue::U16(2),
            IntegerStorage::U16(vec![(u16::MAX - 1) / 2, 3]),
        ),
        (
            IntegerStorage::U32(vec![u32::MAX - 1, 6]),
            IntValue::U32(2),
            IntegerStorage::U32(vec![(u32::MAX - 1) / 2, 3]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX - 1, (1_u64 << 53) + 2]),
            IntValue::U64(2),
            IntegerStorage::U64(vec![(u64::MAX - 1) / 2, (1_u64 << 52) + 1]),
        ),
    ];
    for (array, scalar, expected) in cases {
        let array = integer_tensor(array, vec![1, 2]);
        for divisor in [Value::Int(scalar), Value::Num(2.0)] {
            let result = runmat_runtime::call_builtin("mrdivide", &[array.clone(), divisor])
                .expect("integer scalar-right mrdivide");
            expect_integer(result, &[1, 2], expected.clone());
        }
    }
}

#[test]
fn registered_bit_position_operations_use_scalar_or_exact_size_expansion() {
    let input = integer_tensor(IntegerStorage::U64(vec![1, 1_u64 << 63]), vec![2, 1]);
    let shifts = Value::Tensor(Tensor::new(vec![1.0, -63.0], vec![2, 1]).expect("shifts"));
    expect_integer(
        runmat_runtime::call_builtin("bitshift", &[input.clone(), shifts])
            .expect("same-size bitshift"),
        &[2, 1],
        IntegerStorage::U64(vec![2, 1]),
    );
    let positions = Value::Tensor(Tensor::new(vec![1.0, 64.0], vec![2, 1]).expect("positions"));
    expect_integer(
        runmat_runtime::call_builtin("bitget", &[input.clone(), positions])
            .expect("same-size bitget"),
        &[2, 1],
        IntegerStorage::U64(vec![1, 1]),
    );
    let zeros = integer_tensor(IntegerStorage::U64(vec![0, 0]), vec![2, 1]);
    let positions = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("positions"));
    expect_integer(
        runmat_runtime::call_builtin("bitset", &[zeros, positions, Value::Bool(true)])
            .expect("scalar-expanded bitset value"),
        &[2, 1],
        IntegerStorage::U64(vec![1, 2]),
    );

    let row = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("row"));
    for (builtin, args) in [
        ("bitshift", vec![input.clone(), row.clone()]),
        ("bitget", vec![input.clone(), row.clone()]),
        ("bitset", vec![input.clone(), Value::Num(1.0), row.clone()]),
    ] {
        let error =
            runmat_runtime::call_builtin(builtin, &args).expect_err("singleton expansion rejects");
        assert_eq!(error.identifier(), Some("RunMat:bitwise:SizeMismatch"));
    }
}

#[test]
fn registered_integer_arithmetic_rejects_logicals_before_provider_dispatch() {
    use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

    let integer_handle = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX - 1,
        descriptor: Default::default(),
    };
    let logical_handle = GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    };
    runmat_accelerate_api::set_handle_integer_type(&integer_handle, IntegerElementType::I64);
    runmat_accelerate_api::set_handle_logical(&logical_handle, true);

    let resident_integer = Value::GpuTensor(integer_handle.clone());
    let resident_logical = Value::GpuTensor(logical_handle.clone());
    let host_integer = Value::Int(IntValue::I64(2));
    let host_logical = Value::Bool(true);
    for builtin in [
        "plus", "minus", "times", "rdivide", "ldivide", "power", "rem", "mod",
    ] {
        for (lhs, rhs) in [
            (resident_integer.clone(), host_logical.clone()),
            (host_logical.clone(), resident_integer.clone()),
            (host_integer.clone(), resident_logical.clone()),
            (resident_logical.clone(), host_integer.clone()),
            (resident_integer.clone(), resident_logical.clone()),
            (resident_logical.clone(), resident_integer.clone()),
        ] {
            expect_integer_logical_arithmetic_error(builtin, lhs, rhs);
        }
    }

    runmat_accelerate_api::clear_handle_integer_type(&integer_handle);
    runmat_accelerate_api::clear_handle_logical(&logical_handle);
}

macro_rules! integer_class_cases {
    ($case:ident) => {
        $case!(
            I8,
            vec![2i8, -2],
            vec![3i8, 2],
            vec![5i8, 0],
            vec![-1i8, -4],
            vec![6i8, -4],
            vec![1i8, -1]
        );
        $case!(
            I16,
            vec![2i16, -2],
            vec![3i16, 2],
            vec![5i16, 0],
            vec![-1i16, -4],
            vec![6i16, -4],
            vec![1i16, -1]
        );
        $case!(
            I32,
            vec![2i32, -2],
            vec![3i32, 2],
            vec![5i32, 0],
            vec![-1i32, -4],
            vec![6i32, -4],
            vec![1i32, -1]
        );
        $case!(
            I64,
            vec![2i64, -2],
            vec![3i64, 2],
            vec![5i64, 0],
            vec![-1i64, -4],
            vec![6i64, -4],
            vec![1i64, -1]
        );
        $case!(
            U8,
            vec![2u8, 3],
            vec![3u8, 2],
            vec![5u8, 5],
            vec![0u8, 1],
            vec![6u8, 6],
            vec![1u8, 2]
        );
        $case!(
            U16,
            vec![2u16, 3],
            vec![3u16, 2],
            vec![5u16, 5],
            vec![0u16, 1],
            vec![6u16, 6],
            vec![1u16, 2]
        );
        $case!(
            U32,
            vec![2u32, 3],
            vec![3u32, 2],
            vec![5u32, 5],
            vec![0u32, 1],
            vec![6u32, 6],
            vec![1u32, 2]
        );
        $case!(
            U64,
            vec![2u64, 3],
            vec![3u64, 2],
            vec![5u64, 5],
            vec![0u64, 1],
            vec![6u64, 6],
            vec![1u64, 2]
        );
    };
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_same_class_arithmetic_is_exact_and_preserves_each_class() {
    macro_rules! check {
        ($class:ident, $left:expr, $right:expr, $plus:expr, $minus:expr, $times:expr, $divide:expr) => {{
            let lhs = integer_tensor(IntegerStorage::$class($left), vec![1, 2]);
            let rhs = integer_tensor(IntegerStorage::$class($right), vec![1, 2]);
            for (builtin, expected) in [
                ("plus", IntegerStorage::$class($plus)),
                ("minus", IntegerStorage::$class($minus)),
                ("times", IntegerStorage::$class($times)),
                ("rdivide", IntegerStorage::$class($divide)),
            ] {
                expect_integer(
                    runmat_runtime::call_builtin(builtin, &[lhs.clone(), rhs.clone()])
                        .unwrap_or_else(|error| {
                            panic!("{} {}: {error}", builtin, stringify!($class))
                        }),
                    &[1, 2],
                    expected,
                );
            }
        }};
    }
    integer_class_cases!(check);
}

#[test]
fn integer_arithmetic_saturates_at_every_class_boundary() {
    let cases = [
        (
            IntValue::I8(i8::MAX),
            IntValue::I8(i8::MIN),
            IntValue::I8(i8::MAX),
            IntValue::I8(i8::MIN),
        ),
        (
            IntValue::I16(i16::MAX),
            IntValue::I16(i16::MIN),
            IntValue::I16(i16::MAX),
            IntValue::I16(i16::MIN),
        ),
        (
            IntValue::I32(i32::MAX),
            IntValue::I32(i32::MIN),
            IntValue::I32(i32::MAX),
            IntValue::I32(i32::MIN),
        ),
        (
            IntValue::I64(i64::MAX),
            IntValue::I64(i64::MIN),
            IntValue::I64(i64::MAX),
            IntValue::I64(i64::MIN),
        ),
        (
            IntValue::U8(u8::MAX),
            IntValue::U8(0),
            IntValue::U8(u8::MAX),
            IntValue::U8(0),
        ),
        (
            IntValue::U16(u16::MAX),
            IntValue::U16(0),
            IntValue::U16(u16::MAX),
            IntValue::U16(0),
        ),
        (
            IntValue::U32(u32::MAX),
            IntValue::U32(0),
            IntValue::U32(u32::MAX),
            IntValue::U32(0),
        ),
        (
            IntValue::U64(u64::MAX),
            IntValue::U64(0),
            IntValue::U64(u64::MAX),
            IntValue::U64(0),
        ),
    ];
    for (max, min, saturated_add, saturated_subtract) in cases {
        assert_eq!(
            runmat_runtime::call_builtin("plus", &[Value::Int(max.clone()), Value::Num(1.0)])
                .expect("saturating plus"),
            Value::Int(saturated_add),
        );
        assert_eq!(
            runmat_runtime::call_builtin("minus", &[Value::Int(min), Value::Num(1.0)])
                .expect("saturating minus"),
            Value::Int(saturated_subtract),
        );
    }
}

#[test]
fn mixed_integer_arithmetic_rejects_all_distinct_classes_and_comparisons_remain_exact() {
    let classes = [
        IntValue::I8(0),
        IntValue::I16(0),
        IntValue::I32(0),
        IntValue::I64(0),
        IntValue::U8(0),
        IntValue::U16(0),
        IntValue::U32(0),
        IntValue::U64(0),
    ];
    for (left_index, left) in classes.iter().enumerate() {
        for (right_index, right) in classes.iter().enumerate() {
            if left_index == right_index {
                continue;
            }
            for builtin in ["plus", "minus", "times", "rdivide"] {
                let error = runmat_runtime::call_builtin(
                    builtin,
                    &[Value::Int(left.clone()), Value::Int(right.clone())],
                )
                .expect_err("mixed integer arithmetic must be rejected");
                assert!(
                    error.message().contains("same integer class"),
                    "{builtin}: {error}"
                );
            }
        }
    }

    assert_eq!(
        runmat_runtime::call_builtin(
            "gt",
            &[
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::I64(i64::MAX))
            ]
        )
        .expect("exact mixed comparison"),
        Value::Bool(true),
    );
    assert_eq!(
        runmat_runtime::call_builtin(
            "lt",
            &[Value::Int(IntValue::I64(-1)), Value::Int(IntValue::U64(0))]
        )
        .expect("signed/unsigned comparison"),
        Value::Bool(true),
    );
}

#[test]
fn integer_comparisons_return_logical_arrays_with_broadcast_shape_for_every_class() {
    macro_rules! check_signed {
        ($class:ident, $values:expr) => {{
            let lhs = integer_tensor(IntegerStorage::$class($values), vec![1, 3]);
            let rhs = integer_tensor(IntegerStorage::$class(vec![2, 3, -4]), vec![1, 3]);
            for (builtin, expected) in [
                ("eq", vec![1, 0, 0]),
                ("ne", vec![0, 1, 1]),
                ("lt", vec![0, 0, 0]),
                ("le", vec![1, 0, 0]),
                ("gt", vec![0, 1, 1]),
                ("ge", vec![1, 1, 1]),
            ] {
                expect_logical(
                    runmat_runtime::call_builtin(builtin, &[lhs.clone(), rhs.clone()])
                        .expect("comparison"),
                    &[1, 3],
                    &expected,
                );
            }
        }};
    }
    macro_rules! check_unsigned {
        ($class:ident, $values:expr) => {{
            let lhs = integer_tensor(IntegerStorage::$class($values), vec![1, 3]);
            let rhs = integer_tensor(IntegerStorage::$class(vec![2, 3, 0]), vec![1, 3]);
            for (builtin, expected) in [
                ("eq", vec![1, 0, 0]),
                ("ne", vec![0, 1, 1]),
                ("lt", vec![0, 0, 0]),
                ("le", vec![1, 0, 0]),
                ("gt", vec![0, 1, 1]),
                ("ge", vec![1, 1, 1]),
            ] {
                expect_logical(
                    runmat_runtime::call_builtin(builtin, &[lhs.clone(), rhs.clone()])
                        .expect("comparison"),
                    &[1, 3],
                    &expected,
                );
            }
        }};
    }
    check_signed!(I8, vec![2i8, 5, -2]);
    check_signed!(I16, vec![2i16, 5, -2]);
    check_signed!(I32, vec![2i32, 5, -2]);
    check_signed!(I64, vec![2i64, 5, -2]);
    check_unsigned!(U8, vec![2u8, 5, 2]);
    check_unsigned!(U16, vec![2u16, 5, 2]);
    check_unsigned!(U32, vec![2u32, 5, 2]);
    check_unsigned!(U64, vec![2u64, 5, 2]);
}

#[test]
fn integer_implicit_expansion_appends_singletons_and_enforces_zero_rules_for_every_class() {
    macro_rules! check {
        ($class:ident, $ty:ty) => {{
            let left = integer_tensor(IntegerStorage::$class(vec![1 as $ty, 2 as $ty]), vec![2, 1]);
            let right = integer_tensor(
                IntegerStorage::$class(vec![
                    10 as $ty, 20 as $ty, 30 as $ty, 40 as $ty, 50 as $ty, 60 as $ty,
                ]),
                vec![2, 1, 3],
            );
            expect_integer(
                runmat_runtime::call_builtin("plus", &[left, right])
                    .expect("rank-mismatched integer expansion"),
                &[2, 1, 3],
                IntegerStorage::$class(vec![
                    11 as $ty, 22 as $ty, 31 as $ty, 42 as $ty, 51 as $ty, 62 as $ty,
                ]),
            );

            let row_shorthand = integer_tensor(
                IntegerStorage::$class(vec![1 as $ty, 2 as $ty, 3 as $ty]),
                vec![3],
            );
            let pages = integer_tensor(
                IntegerStorage::$class(vec![
                    10 as $ty, 20 as $ty, 30 as $ty, 40 as $ty, 50 as $ty, 60 as $ty,
                ]),
                vec![1, 3, 2],
            );
            expect_integer(
                runmat_runtime::call_builtin("plus", &[row_shorthand, pages])
                    .expect("one-dimensional row shorthand expansion"),
                &[1, 3, 2],
                IntegerStorage::$class(vec![
                    11 as $ty, 22 as $ty, 33 as $ty, 41 as $ty, 52 as $ty, 63 as $ty,
                ]),
            );

            let old_front_aligned_left = integer_tensor(
                IntegerStorage::$class(vec![
                    1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty, 5 as $ty, 6 as $ty,
                ]),
                vec![2, 3],
            );
            let old_front_aligned_right = integer_tensor(
                IntegerStorage::$class(vec![
                    1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty, 5 as $ty, 6 as $ty,
                ]),
                vec![1, 2, 3],
            );
            let rank_error = runmat_runtime::call_builtin(
                "plus",
                &[old_front_aligned_left, old_front_aligned_right],
            )
            .expect_err("front-aligned rank expansion must be rejected");
            assert!(
                rank_error.message().contains("dimension mismatch"),
                "unexpected rank mismatch error: {rank_error}"
            );

            let empty = integer_tensor(IntegerStorage::$class(Vec::<$ty>::new()), vec![0, 3]);
            let singleton = integer_tensor(
                IntegerStorage::$class(vec![1 as $ty, 2 as $ty, 3 as $ty]),
                vec![1, 3],
            );
            expect_integer(
                runmat_runtime::call_builtin("plus", &[empty.clone(), singleton])
                    .expect("zero extent expands against singleton"),
                &[0, 3],
                IntegerStorage::$class(Vec::<$ty>::new()),
            );

            let nonsingleton = integer_tensor(
                IntegerStorage::$class(vec![
                    1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty, 5 as $ty, 6 as $ty,
                ]),
                vec![2, 3],
            );
            let zero_error = runmat_runtime::call_builtin("plus", &[empty, nonsingleton])
                .expect_err("zero extent cannot expand against a nonsingleton");
            assert!(
                zero_error.message().contains("dimension mismatch"),
                "unexpected zero mismatch error: {zero_error}"
            );
        }};
    }

    check!(I8, i8);
    check!(I16, i16);
    check!(I32, i32);
    check!(I64, i64);
    check!(U8, u8);
    check!(U16, u16);
    check!(U32, u32);
    check!(U64, u64);
}

#[test]
fn native_integer_reductions_preserve_class_shape_and_empty_identities() {
    macro_rules! check {
        ($class:ident, $values:expr, $sum:expr, $product:expr, $mean:expr, $zero:expr, $one:expr) => {{
            let matrix = integer_tensor(IntegerStorage::$class($values), vec![2, 2]);
            for (builtin, expected) in [
                ("sum", IntegerStorage::$class($sum)),
                ("prod", IntegerStorage::$class($product)),
                ("mean", IntegerStorage::$class($mean)),
            ] {
                expect_integer(
                    runmat_runtime::call_builtin(builtin, &[matrix.clone(), Value::from("native")])
                        .expect("native reduction"),
                    &[1, 2],
                    expected,
                );
            }
            let scalar = Value::Int(IntValue::$class($zero));
            assert_eq!(
                runmat_runtime::call_builtin("sum", &[scalar.clone(), Value::from("native")])
                    .expect("scalar sum"),
                scalar
            );
            assert_eq!(
                runmat_runtime::call_builtin("prod", &[scalar.clone(), Value::from("native")])
                    .expect("scalar prod"),
                scalar
            );

            let empty = integer_tensor(IntegerStorage::$class(Vec::new()), vec![0, 2]);
            expect_integer(
                runmat_runtime::call_builtin("sum", &[empty.clone(), Value::from("native")])
                    .expect("empty sum"),
                &[1, 2],
                IntegerStorage::$class(vec![$zero, $zero]),
            );
            expect_integer(
                runmat_runtime::call_builtin("prod", &[empty, Value::from("native")])
                    .expect("empty product"),
                &[1, 2],
                IntegerStorage::$class(vec![$one, $one]),
            );
        }};
    }
    check!(
        I8,
        vec![2i8, 3, 4, 5],
        vec![5i8, 9],
        vec![6i8, 20],
        vec![3i8, 5],
        0i8,
        1i8
    );
    check!(
        I16,
        vec![2i16, 3, 4, 5],
        vec![5i16, 9],
        vec![6i16, 20],
        vec![3i16, 5],
        0i16,
        1i16
    );
    check!(
        I32,
        vec![2i32, 3, 4, 5],
        vec![5i32, 9],
        vec![6i32, 20],
        vec![3i32, 5],
        0i32,
        1i32
    );
    check!(
        I64,
        vec![2i64, 3, 4, 5],
        vec![5i64, 9],
        vec![6i64, 20],
        vec![3i64, 5],
        0i64,
        1i64
    );
    check!(
        U8,
        vec![2u8, 3, 4, 5],
        vec![5u8, 9],
        vec![6u8, 20],
        vec![3u8, 5],
        0u8,
        1u8
    );
    check!(
        U16,
        vec![2u16, 3, 4, 5],
        vec![5u16, 9],
        vec![6u16, 20],
        vec![3u16, 5],
        0u16,
        1u16
    );
    check!(
        U32,
        vec![2u32, 3, 4, 5],
        vec![5u32, 9],
        vec![6u32, 20],
        vec![3u32, 5],
        0u32,
        1u32
    );
    check!(
        U64,
        vec![2u64, 3, 4, 5],
        vec![5u64, 9],
        vec![6u64, 20],
        vec![3u64, 5],
        0u64,
        1u64
    );
}

#[test]
fn integer_elementwise_extrema_preserve_all_classes_and_broadcast_exactly() {
    macro_rules! check {
        ($class:ident, $left:expr, $right:expr, $min:expr, $max:expr) => {{
            let left = integer_tensor(IntegerStorage::$class($left), vec![2, 1]);
            let right = integer_tensor(IntegerStorage::$class($right), vec![1, 2]);
            expect_integer(
                runmat_runtime::call_builtin("min", &[left.clone(), right.clone()])
                    .expect("integer min"),
                &[2, 2],
                IntegerStorage::$class($min),
            );
            expect_integer(
                runmat_runtime::call_builtin("max", &[left, right]).expect("integer max"),
                &[2, 2],
                IntegerStorage::$class($max),
            );
        }};
    }
    check!(
        I8,
        vec![-3i8, 4],
        vec![2i8, -5],
        vec![-3i8, 2, -5, -5],
        vec![2i8, 4, -3, 4]
    );
    check!(
        I16,
        vec![-3i16, 4],
        vec![2i16, -5],
        vec![-3i16, 2, -5, -5],
        vec![2i16, 4, -3, 4]
    );
    check!(
        I32,
        vec![-3i32, 4],
        vec![2i32, -5],
        vec![-3i32, 2, -5, -5],
        vec![2i32, 4, -3, 4]
    );
    check!(
        I64,
        vec![i64::MIN, 4],
        vec![-3i64, i64::MAX],
        vec![i64::MIN, -3, i64::MIN, 4],
        vec![-3i64, 4, i64::MAX, i64::MAX]
    );
    check!(
        U8,
        vec![3u8, 4],
        vec![2u8, 5],
        vec![2u8, 2, 3, 4],
        vec![3u8, 4, 5, 5]
    );
    check!(
        U16,
        vec![3u16, 4],
        vec![2u16, 5],
        vec![2u16, 2, 3, 4],
        vec![3u16, 4, 5, 5]
    );
    check!(
        U32,
        vec![3u32, 4],
        vec![2u32, 5],
        vec![2u32, 2, 3, 4],
        vec![3u32, 4, 5, 5]
    );
    check!(
        U64,
        vec![(1_u64 << 53) + 1, u64::MAX],
        vec![1_u64 << 53, u64::MAX - 1],
        vec![1_u64 << 53, 1_u64 << 53, (1_u64 << 53) + 1, u64::MAX - 1],
        vec![(1_u64 << 53) + 1, u64::MAX, u64::MAX - 1, u64::MAX]
    );
}

#[test]
fn integer_logical_reductions_and_nnz_use_typed_storage_for_all_classes() {
    macro_rules! check {
        ($class:ident, $values:expr) => {{
            let tensor = Tensor::new_integer(IntegerStorage::$class($values), vec![2, 2])
                .expect("integer tensor");
            let value = Value::Tensor(tensor);
            expect_logical(
                runmat_runtime::call_builtin("any", std::slice::from_ref(&value)).expect("any"),
                &[1, 2],
                &[1, 1],
            );
            expect_logical(
                runmat_runtime::call_builtin("all", std::slice::from_ref(&value)).expect("all"),
                &[1, 2],
                &[0, 1],
            );
            assert_eq!(
                runmat_runtime::call_builtin("nnz", &[value]).expect("nnz"),
                Value::Num(3.0),
            );
        }};
    }
    check!(I8, vec![0i8, -1, 2, 3]);
    check!(I16, vec![0i16, -1, 2, 3]);
    check!(I32, vec![0i32, -1, 2, 3]);
    check!(I64, vec![0i64, i64::MIN, 2, 3]);
    check!(U8, vec![0u8, 1, 2, 3]);
    check!(U16, vec![0u16, 1, 2, 3]);
    check!(U32, vec![0u32, 1, 2, 3]);
    check!(U64, vec![0u64, (1_u64 << 53) + 1, 2, u64::MAX]);
}

#[test]
fn registered_set_functions_cover_all_ordered_integer_class_pairs_and_rows() {
    let classes = integer_set_classes();
    let builtins = [
        ("ismember", "RunMat:ismember:NumericClassMismatch"),
        ("intersect", "RunMat:intersect:NumericClassMismatch"),
        ("union", "RunMat:union:NumericClassMismatch"),
        ("setdiff", "RunMat:setdiff:NumericClassMismatch"),
        ("setxor", "RunMat:setxor:NumericClassMismatch"),
    ];
    for (left_index, left_storage) in classes.iter().enumerate() {
        for (right_index, right_storage) in classes.iter().enumerate() {
            for (builtin, mismatch_identifier) in builtins {
                for rows in [false, true] {
                    let left = integer_tensor(left_storage.clone(), vec![2, 2]);
                    let right = integer_tensor(right_storage.clone(), vec![2, 2]);
                    let mut args = vec![left, right];
                    if rows {
                        args.push(Value::from("rows"));
                    }
                    if left_index == right_index {
                        let result = runmat_runtime::call_builtin(builtin, &args)
                            .unwrap_or_else(|error| panic!("{builtin} same-class call: {error}"));
                        if builtin == "ismember" {
                            assert!(matches!(result, Value::Bool(_) | Value::LogicalArray(_)));
                        } else {
                            expect_integer_class(result, left_storage.class_name());
                        }
                    } else {
                        let error = runmat_runtime::call_builtin(builtin, &args)
                            .expect_err("unlike nondouble integer classes must reject");
                        assert_eq!(
                            error.identifier(),
                            Some(mismatch_identifier),
                            "{builtin} pair {} / {} in {} mode",
                            left_storage.class_name(),
                            right_storage.class_name(),
                            if rows { "rows" } else { "elements" }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn registered_set_functions_apply_double_exception_and_index_classes_for_every_integer() {
    let classes = integer_set_classes();
    for storage in classes {
        let class_name = storage.class_name();
        let integer = integer_tensor(storage, vec![4, 1]);
        let double =
            Value::Tensor(Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![4, 1]).expect("double"));

        for builtin in ["intersect", "union", "setxor"] {
            for args in [
                [integer.clone(), double.clone()],
                [double.clone(), integer.clone()],
            ] {
                expect_integer_class(
                    runmat_runtime::call_builtin(builtin, &args)
                        .unwrap_or_else(|error| panic!("{builtin} double exception: {error}")),
                    class_name,
                );
            }
        }

        expect_integer_class(
            runmat_runtime::call_builtin("setdiff", &[integer.clone(), double.clone()])
                .expect("integer-left setdiff"),
            class_name,
        );
        expect_double_numeric(
            &runmat_runtime::call_builtin("setdiff", &[double.clone(), integer.clone()])
                .expect("double-left setdiff"),
        );

        for args in [
            [integer.clone(), double.clone()],
            [double.clone(), integer.clone()],
        ] {
            let result = futures::executor::block_on(
                runmat_runtime::call_builtin_async_with_outputs("ismember", &args, 2),
            )
            .expect("ismember double exception");
            let Value::OutputList(outputs) = result else {
                panic!("ismember multi-output result")
            };
            assert!(matches!(
                outputs.first(),
                Some(Value::Bool(_) | Value::LogicalArray(_))
            ));
            expect_double_numeric(&outputs[1]);
        }
    }

    let integer = integer_tensor(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![4, 1]);
    for (builtin, output_count) in [
        ("intersect", 3),
        ("union", 3),
        ("setdiff", 2),
        ("setxor", 3),
    ] {
        let result = futures::executor::block_on(runmat_runtime::call_builtin_async_with_outputs(
            builtin,
            &[integer.clone(), integer.clone()],
            output_count,
        ))
        .expect("set index outputs");
        let Value::OutputList(outputs) = result else {
            panic!("{builtin} multi-output result")
        };
        expect_integer_class(outputs[0].clone(), "uint64");
        for output in &outputs[1..] {
            expect_double_numeric(output);
        }
    }
}

#[test]
fn registered_set_double_exception_keeps_wide_integer_comparison_exact() {
    let wide = (1_u64 << 53) + 1;
    let rounded = 1_u64 << 53;
    assert_eq!(wide as f64, rounded as f64);
    let integer = integer_tensor(IntegerStorage::U64(vec![wide]), vec![1, 1]);
    let double = Value::Num(rounded as f64);

    let member = runmat_runtime::call_builtin("ismember", &[integer.clone(), double.clone()])
        .expect("wide ismember");
    assert_eq!(member, Value::Bool(false));

    expect_integer(
        runmat_runtime::call_builtin("intersect", &[integer.clone(), double.clone()])
            .expect("wide intersect"),
        &[0, 1],
        IntegerStorage::U64(Vec::new()),
    );
    expect_integer(
        runmat_runtime::call_builtin("union", &[integer.clone(), double.clone()])
            .expect("wide union"),
        &[2, 1],
        IntegerStorage::U64(vec![rounded, wide]),
    );
    expect_integer(
        runmat_runtime::call_builtin("setdiff", &[integer.clone(), double.clone()])
            .expect("wide setdiff"),
        &[1, 1],
        IntegerStorage::U64(vec![wide]),
    );
    expect_integer(
        runmat_runtime::call_builtin("setxor", &[integer, double]).expect("wide setxor"),
        &[1, 2],
        IntegerStorage::U64(vec![rounded, wide]),
    );
}

#[test]
fn registered_set_functions_reject_64_bit_integer_gpu_inputs_before_provider_dispatch() {
    use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

    for (offset, integer_type) in [IntegerElementType::I64, IntegerElementType::U64]
        .into_iter()
        .enumerate()
    {
        let handle = GpuTensorHandle {
            shape: vec![2, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 100 - offset as u64,
            descriptor: Default::default(),
        };
        runmat_accelerate_api::set_handle_integer_type(&handle, integer_type);
        let resident = Value::GpuTensor(handle.clone());
        let host = integer_tensor(IntegerStorage::I32(vec![1, 2]), vec![2, 1]);
        for builtin in ["ismember", "intersect", "union", "setdiff", "setxor"] {
            for args in [
                [resident.clone(), host.clone()],
                [host.clone(), resident.clone()],
                [resident.clone(), resident.clone()],
            ] {
                let error = runmat_runtime::call_builtin(builtin, &args)
                    .expect_err("64-bit integer gpuArray must reject");
                assert!(
                    error.identifier().is_some_and(|identifier| {
                        identifier.ends_with(":UnsupportedInputType")
                    }),
                    "{builtin}: {error}"
                );
                assert!(
                    error
                        .message()
                        .contains("resident 64-bit integer inputs are not supported"),
                    "{builtin}: {error}"
                );
            }
        }
        runmat_accelerate_api::clear_handle_integer_type(&handle);
    }
}
