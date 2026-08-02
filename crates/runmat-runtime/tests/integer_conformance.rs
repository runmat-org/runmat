#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use runmat_builtins::{IntValue, IntegerStorage, LogicalArray, Tensor, Value};

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

fn expect_logical(value: Value, shape: &[usize], data: &[u8]) {
    assert_eq!(
        value,
        Value::LogicalArray(LogicalArray::new(data.to_vec(), shape.to_vec()).expect("logical"))
    );
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
