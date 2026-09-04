use futures::executor::block_on;
use runmat_value::{CharArray, IntValue, IntegerStorage, NumericDType, Tensor, Value};

use super::*;

fn tensor(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).expect("valid test tensor"))
}

fn empty() -> Value {
    tensor(Vec::new(), vec![0, 0])
}

#[test]
fn sums_vector_and_matrix_subscripts() {
    let vector = block_on(accumarray_builtin(
        tensor(vec![1.0, 3.0, 4.0, 2.0, 4.0, 1.0], vec![6, 1]),
        tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]),
        Vec::new(),
    ))
    .expect("vector accumulation");
    let Value::Tensor(vector) = vector else {
        panic!("expected tensor");
    };
    assert_eq!(vector.materialize_f64(), vec![7.0, 4.0, 2.0, 8.0]);

    let matrix = block_on(accumarray_builtin(
        tensor(
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
            vec![6, 2],
        ),
        tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]),
        Vec::new(),
    ))
    .expect("matrix accumulation");
    let Value::Tensor(matrix) = matrix else {
        panic!("expected tensor");
    };
    assert_eq!(matrix.shape, vec![4, 2]);
    assert_eq!(
        matrix.materialize_f64(),
        vec![5.0, 0.0, 0.0, 6.0, 0.0, 7.0, 3.0, 0.0]
    );
}

#[test]
fn preserves_callback_numeric_classes() {
    for storage in integer_triplets() {
        let fill = storage.zeros_like(1).value_at(0).expect("integer zero");
        let expected = storage
            .from_exact_values_like(vec![
                storage.value_at(1).unwrap(),
                storage.value_at(2).unwrap(),
                fill.clone(),
            ])
            .expect("same-class values");
        let output = block_on(accumarray_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 2]), vec![3, 1]).unwrap(),
            ),
            Value::Tensor(Tensor::new_integer(storage, vec![3, 1]).unwrap()),
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![3, 1]), vec![1, 2]).unwrap(),
                ),
                Value::FunctionHandle("min".into()),
                Value::Int(fill),
            ],
        ))
        .expect("integer callback accumulation");
        let Value::Tensor(output) = output else {
            panic!("expected integer tensor")
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }

    let single = Value::Tensor(
        Tensor::new_with_dtype(vec![4.0, 2.0, 9.0], vec![3, 1], NumericDType::F32).unwrap(),
    );
    let output = block_on(accumarray_builtin(
        tensor(vec![1.0, 1.0, 2.0], vec![3, 1]),
        single,
        vec![
            tensor(vec![3.0, 1.0], vec![1, 2]),
            Value::FunctionHandle("min".into()),
        ],
    ))
    .expect("single callback accumulation");
    let Value::Tensor(output) = output else {
        panic!("expected single tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn sparse_output_enforces_compatible_constraints() {
    let sparse = block_on(accumarray_builtin(
        tensor(vec![1.0, 3.0], vec![2, 1]),
        Value::Num(1.0),
        vec![
            tensor(vec![4.0, 1.0], vec![1, 2]),
            empty(),
            empty(),
            Value::Bool(true),
        ],
    ))
    .expect("sparse accumulation");
    let Value::SparseTensor(sparse) = sparse else {
        panic!("expected sparse tensor")
    };
    assert_eq!((sparse.rows, sparse.cols, sparse.nnz()), (4, 1, 2));

    let integer_data =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap());
    let error = block_on(accumarray_builtin(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        integer_data,
        vec![empty(), empty(), empty(), Value::Bool(true)],
    ))
    .expect_err("sparse integer data must fail");
    assert!(error.message.contains("requires double input data"));
}

#[test]
fn rejects_invalid_indices_sizes_and_controls() {
    let error = block_on(accumarray_builtin(
        tensor(vec![1.0, -1.0], vec![2, 1]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Vec::new(),
    ))
    .expect_err("negative index must fail");
    assert!(error.message.contains("positive integer"));

    let error = block_on(accumarray_builtin(
        Value::Num(1.0),
        Value::Num(1.0),
        vec![empty(), empty(), empty(), Value::Int(IntValue::U8(2))],
    ))
    .expect_err("nonbinary sparse control must fail");
    assert!(error.message.contains("numeric 0 or 1"));
}

#[test]
fn scalar_character_results_materialize_as_a_character_array() {
    let result = output::from_scalars(
        vec![
            Value::CharArray(CharArray::new_row("a")),
            Value::CharArray(CharArray::new_row("b")),
        ],
        vec![2, 1],
        false,
    )
    .expect("character accumulation output");
    let Value::CharArray(result) = result else {
        panic!("expected character array");
    };
    assert_eq!(result.to_column_major(), vec!['a', 'b']);
    assert_eq!(result.shape, vec![2, 1]);
}

#[test]
fn every_integer_class_is_exact_for_indices_and_sizes() {
    for storage in integer_pairs() {
        let size = storage
            .from_exact_values_like(vec![storage.cast_f64_assignment(3.0)])
            .expect("same-class size");
        let output = block_on(accumarray_builtin(
            Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap()),
            Value::Num(1.0),
            vec![Value::Tensor(
                Tensor::new_integer(size, vec![1, 1]).unwrap(),
            )],
        ))
        .expect("integer structural inputs");
        let Value::Tensor(output) = output else {
            panic!("expected dense output");
        };
        assert_eq!(output.shape, vec![3, 1]);
        assert_eq!(output.materialize_f64(), vec![1.0, 1.0, 0.0]);
    }
}

#[test]
fn resident_integer_data_is_rejected_before_provider_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let data = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &data).unwrap();
        let error = block_on(accumarray_builtin(
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        ))
        .expect_err("resident integer data must reject");
        assert!(error.message.contains("GPU input data must be"));
        provider.free(&handle).expect("free test allocation");
    });
}

fn integer_triplets() -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![4, 2, 9]),
        IntegerStorage::I16(vec![4, 2, 9]),
        IntegerStorage::I32(vec![4, 2, 9]),
        IntegerStorage::I64(vec![4, 2, 9]),
        IntegerStorage::U8(vec![4, 2, 9]),
        IntegerStorage::U16(vec![4, 2, 9]),
        IntegerStorage::U32(vec![4, 2, 9]),
        IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1, 9]),
    ]
}

fn integer_pairs() -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![1, 2]),
        IntegerStorage::I16(vec![1, 2]),
        IntegerStorage::I32(vec![1, 2]),
        IntegerStorage::I64(vec![1, 2]),
        IntegerStorage::U8(vec![1, 2]),
        IntegerStorage::U16(vec![1, 2]),
        IntegerStorage::U32(vec![1, 2]),
        IntegerStorage::U64(vec![1, 2]),
    ]
}
