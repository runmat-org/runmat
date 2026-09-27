use futures::executor::block_on;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, NumericElementType};
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::{CharArray, NumericStorage, Tensor, Value};

fn array(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn invoke(args: &[Value], outputs: usize) -> Result<Vec<Value>, runmat_runtime::RuntimeError> {
    block_on(call_builtin_async_with_outputs("discretize", args, outputs)).map(
        |value| match value {
            Value::OutputList(values) => values,
            value => vec![value],
        },
    )
}

#[test]
fn numeric_bin_admission_precedes_any_resident_download() {
    let poisoned = Value::GpuTensor(
        GpuTensorHandle::new(vec![1, 2], u32::MAX, u64::MAX)
            .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real),
    );
    let edges = array(vec![0.0, 1.0, 2.0], vec![1, 3]);
    let cases = [
        (vec![poisoned.clone(), edges.clone()], 3),
        (vec![poisoned.clone(), edges.clone()], 2),
        (
            vec![
                poisoned.clone(),
                edges.clone(),
                Value::from("IncludedEdge"),
                Value::from("middle"),
            ],
            0,
        ),
        (vec![poisoned.clone(), array(vec![1.0, 0.0], vec![1, 2])], 1),
        (
            vec![
                poisoned.clone(),
                array(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            ],
            1,
        ),
        (
            vec![poisoned, edges, array(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
            1,
        ),
    ];
    for (args, outputs) in cases {
        let error = invoke(&args, outputs).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:discretize:InvalidInput"));
    }
}

#[test]
fn scalar_tensor_counts_and_extreme_edges_preserve_shape_and_values() {
    for storage in [NumericStorage::F32(vec![2.0]), NumericStorage::U64(vec![2])] {
        let args = [
            array(vec![-f64::MAX, 0.0, f64::MAX], vec![1, 1, 3]),
            Value::Tensor(Tensor::from_numeric_storage(storage, vec![1, 1, 1]).unwrap()),
        ];
        let output = invoke(&args, 2).unwrap();
        let Value::Tensor(indices) = &output[0] else {
            panic!("dense indices")
        };
        let Value::Tensor(edges) = &output[1] else {
            panic!("dense edges")
        };
        assert_eq!(indices.shape, vec![1, 1, 3]);
        assert_eq!(indices.materialize_f64(), vec![1.0, 2.0, 2.0]);
        assert_eq!(edges.materialize_f64(), vec![-f64::MAX, 0.0, f64::MAX]);
        assert!(invoke(&args, 0).unwrap().is_empty());
    }
}

#[test]
fn text_options_use_effective_rank_and_scalar_string_arrays() {
    let input = array(vec![1.0], vec![1, 1]);
    let edges = array(vec![0.0, 1.0, 2.0], vec![1, 3, 1]);
    let option = Value::CharArray(
        CharArray::new_with_shape("IncludedEdge".chars().collect(), vec![1, 12, 1]).unwrap(),
    );
    let output = invoke(
        &[input.clone(), edges.clone(), option, Value::from("right")],
        1,
    )
    .unwrap();
    let Value::Tensor(output) = &output[0] else {
        panic!("dense scalar output")
    };
    assert_eq!(output.materialize_f64(), vec![1.0]);
    let pages = Value::CharArray(
        CharArray::new_with_shape("IncludedEdge".chars().collect(), vec![1, 6, 2]).unwrap(),
    );
    assert!(invoke(&[input, edges, pages, Value::from("right")], 1).is_err());
}
