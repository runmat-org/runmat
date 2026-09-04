use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::{NEXTPOW2_ERROR_INVALID_INPUT, NEXTPOW2_ERROR_TOO_MANY_OUTPUTS};
use runmat_value::{IntegerStorage, NumericStorage, Tensor};

#[test]
fn scalars_zero_and_special_values_follow_the_numeric_contract() {
    for (input, expected) in [(9.0, 4.0), (0.0, 0.0), (-3.0, 2.0)] {
        assert_eq!(
            block_on(nextpow2_builtin(Value::Num(input))).expect("nextpow2"),
            Value::Num(expected)
        );
    }
    let Value::Num(infinite) =
        block_on(nextpow2_builtin(Value::Num(f64::INFINITY))).expect("infinity")
    else {
        panic!("expected scalar");
    };
    assert!(infinite.is_infinite());
    let Value::Num(nan) = block_on(nextpow2_builtin(Value::Num(f64::NAN))).expect("NaN") else {
        panic!("expected scalar");
    };
    assert!(nan.is_nan());
}

#[test]
fn tensors_preserve_shape_and_floating_class() {
    let double = Tensor::new(vec![0.0, 1.0, 3.0, 9.0], vec![2, 2]).expect("double tensor");
    let Value::Tensor(output) =
        block_on(nextpow2_builtin(Value::Tensor(double))).expect("double output")
    else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 2.0, 4.0]);

    let single = Tensor::from_f32(vec![0.0, -3.0, 9.0], vec![1, 3]).expect("single tensor");
    let Value::Tensor(output) =
        block_on(nextpow2_builtin(Value::Tensor(single))).expect("single output")
    else {
        panic!("expected single tensor");
    };
    assert_eq!(
        output.into_numeric_storage().expect("numeric storage"),
        NumericStorage::F32(vec![0.0, 2.0, 4.0])
    );

    let empty = Tensor::from_f32(Vec::new(), vec![0, 3]).expect("empty tensor");
    let Value::Tensor(output) =
        block_on(nextpow2_builtin(Value::Tensor(empty))).expect("empty output")
    else {
        panic!("expected empty tensor");
    };
    assert_eq!(output.shape, vec![0, 3]);
    assert_eq!(
        output.into_numeric_storage().expect("numeric storage"),
        NumericStorage::F32(Vec::new())
    );
}

#[test]
fn every_integer_class_preserves_exact_storage() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MIN, -3, 0, 9]),
            IntegerStorage::I8(vec![7, 2, 0, 4]),
        ),
        (
            IntegerStorage::I16(vec![i16::MIN, -3, 0, 9]),
            IntegerStorage::I16(vec![15, 2, 0, 4]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, -3, 0, 9]),
            IntegerStorage::I32(vec![31, 2, 0, 4]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, -3, 0, 9]),
            IntegerStorage::I64(vec![63, 2, 0, 4]),
        ),
        (
            IntegerStorage::U8(vec![0, 1, 3, u8::MAX]),
            IntegerStorage::U8(vec![0, 0, 2, 8]),
        ),
        (
            IntegerStorage::U16(vec![0, 1, 3, u16::MAX]),
            IntegerStorage::U16(vec![0, 0, 2, 16]),
        ),
        (
            IntegerStorage::U32(vec![0, 1, 3, u32::MAX]),
            IntegerStorage::U32(vec![0, 0, 2, 32]),
        ),
        (
            IntegerStorage::U64(vec![0, 1, 3, u64::MAX]),
            IntegerStorage::U64(vec![0, 0, 2, 64]),
        ),
    ];
    for (input, expected) in cases {
        let tensor = Tensor::new_integer(input, vec![1, 4]).expect("integer tensor");
        let Value::Tensor(output) =
            block_on(nextpow2_builtin(Value::Tensor(tensor))).expect("integer output")
        else {
            panic!("expected integer tensor");
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}

#[test]
fn logical_input_promotes_to_double() {
    let input = runmat_value::LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical");
    let Value::Tensor(output) =
        block_on(nextpow2_builtin(Value::LogicalArray(input))).expect("logical output")
    else {
        panic!("expected tensor");
    };
    assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 0.0]);
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
}

#[test]
fn provider_results_preserve_class_shape_and_residency() {
    test_support::with_test_provider(|provider| {
        for (tensor, expected) in [
            (
                Tensor::new(vec![0.0, 1.0, 3.0, 9.0], vec![4, 1]).expect("double"),
                NumericStorage::F64(vec![0.0, 0.0, 2.0, 4.0]),
            ),
            (
                Tensor::new_integer(IntegerStorage::U64(vec![0, 3, u64::MAX]), vec![1, 3])
                    .expect("integer"),
                NumericStorage::U64(vec![0, 2, 64]),
            ),
        ] {
            let expected_class = tensor.numeric_dtype();
            let expected_shape = tensor.shape.clone();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = block_on(nextpow2_builtin(Value::GpuTensor(handle))).expect("provider");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.numeric_dtype(), expected_class);
            assert_eq!(gathered.shape, expected_shape);
            assert_eq!(
                gathered.into_numeric_storage().expect("numeric storage"),
                expected
            );
        }
    });
}

#[test]
fn unsupported_input_and_excess_output_have_stable_identifiers() {
    let error = block_on(nextpow2_builtin(Value::from("bad"))).expect_err("invalid input");
    assert_eq!(error.identifier(), NEXTPOW2_ERROR_INVALID_INPUT.identifier);

    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(nextpow2_builtin(Value::Num(2.0))).expect_err("too many outputs");
    assert_eq!(
        error.identifier(),
        NEXTPOW2_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}
